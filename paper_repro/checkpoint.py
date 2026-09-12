from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.distributed.checkpoint import load as dcp_load
from torch.distributed.checkpoint import save as dcp_save
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_state_dict,
)

from paper_repro.common import atomic_json_dump, sha256_file
from paper_repro.fsdp2 import ParallelContext, reshard_all


def _sharded_options() -> StateDictOptions:
    return StateDictOptions(
        full_state_dict=False,
        cpu_offload=False,
        ignore_frozen_params=False,
        strict=True,
        broadcast_from_rank0=False,
    )


def _rank_zero_full_options() -> StateDictOptions:
    return StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
        ignore_frozen_params=False,
        strict=True,
        broadcast_from_rank0=False,
    )


def initialize_adamw_state(optimizer: torch.optim.AdamW) -> None:
    """Materialize zero AdamW state without taking an optimizer step.

    PyTorch's distributed state-dict loader initializes every optimizer entry
    before planning a strict load. DiffuSETS has several declared parameters
    that are never reached by ``forward``; AdamW would otherwise leave their
    state absent from the saved checkpoint and a strict resume would fail.
    This mirrors AdamW's own lazy initialization with a step value of zero, so
    the first real update retains the released optimizer semantics.
    """

    for group in optimizer.param_groups:
        capturable = bool(group["capturable"])
        fused = bool(group["fused"])
        for parameter in group["params"]:
            if not parameter.requires_grad or optimizer.state[parameter]:
                continue
            state = optimizer.state[parameter]
            step_device = (
                parameter.device if capturable or fused else torch.device("cpu")
            )
            state["step"] = torch.zeros((), dtype=torch.float32, device=step_device)
            state["exp_avg"] = torch.zeros_like(
                parameter, memory_format=torch.preserve_format
            )
            state["exp_avg_sq"] = torch.zeros_like(
                parameter, memory_format=torch.preserve_format
            )
            if group["amsgrad"]:
                state["max_exp_avg_sq"] = torch.zeros_like(
                    parameter, memory_format=torch.preserve_format
                )


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state(state["torch_cuda"])


def save_resume_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    output_dir: Path,
    epoch: int,
    global_step: int,
    parallel: ParallelContext,
    provenance: dict[str, Any],
) -> Path:
    """Save a reshardable FSDP2 model/optimizer checkpoint with rank RNG."""

    reshard_all(model)
    target = output_dir / "checkpoints" / f"epoch_{epoch:03d}"
    temporary = target.with_name(f".{target.name}.tmp")
    if parallel.is_rank_zero:
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
    parallel.barrier()

    model_state, optimizer_state = get_state_dict(
        model, optimizer, options=_sharded_options()
    )
    dcp_save(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id=temporary / "dcp",
    )
    torch.save(capture_rng_state(), temporary / f"rng_rank_{parallel.rank:04d}.pt")
    parallel.barrier()
    if parallel.is_rank_zero:
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
            },
            temporary / "trainer_state.pt",
        )
        atomic_json_dump(provenance, temporary / "provenance.json")
        if target.exists():
            shutil.rmtree(target)
        os.replace(temporary, target)
        atomic_json_dump(
            {
                "epoch": epoch,
                "global_step": global_step,
                "checkpoint": str(target),
                "provenance_sha256": sha256_file(target / "provenance.json"),
            },
            output_dir / "latest.json",
        )
    parallel.barrier()
    return target


def load_resume_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    checkpoint: Path,
    parallel: ParallelContext,
) -> tuple[int, int, dict[str, Any]]:
    trainer_path = checkpoint / "trainer_state.pt"
    provenance_path = checkpoint / "provenance.json"
    if not trainer_path.exists() or not provenance_path.exists():
        raise FileNotFoundError(f"incomplete FSDP2 checkpoint: {checkpoint}")
    with provenance_path.open(encoding="utf-8") as handle:
        provenance = json.load(handle)
    if provenance.get("world_size") != parallel.world_size:
        raise ValueError(
            "exact RNG resume requires the checkpoint world size "
            f"{provenance.get('world_size')} (current {parallel.world_size})"
        )
    model_state, optimizer_state = get_state_dict(
        model, optimizer, options=_sharded_options()
    )
    state = {"model": model_state, "optimizer": optimizer_state}
    dcp_load(state, checkpoint_id=checkpoint / "dcp")
    incompatible = set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
        options=_sharded_options(),
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"incompatible checkpoint keys: {incompatible}")
    trainer = torch.load(trainer_path, map_location="cpu", weights_only=False)
    if scheduler is not None and trainer["scheduler"] is not None:
        scheduler.load_state_dict(trainer["scheduler"])
    rng_path = checkpoint / f"rng_rank_{parallel.rank:04d}.pt"
    if not rng_path.exists():
        raise FileNotFoundError(f"checkpoint lacks RNG state for rank {parallel.rank}")
    restore_rng_state(torch.load(rng_path, map_location="cpu", weights_only=False))
    parallel.barrier()
    return int(trainer["epoch"]), int(trainer["global_step"]), provenance


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    pointer = output_dir / "latest.json"
    if not pointer.exists():
        return None
    with pointer.open(encoding="utf-8") as handle:
        value = json.load(handle)
    path = Path(value["checkpoint"])
    if not path.exists():
        raise FileNotFoundError(
            f"latest checkpoint pointer targets missing path {path}"
        )
    return path


def save_portable_checkpoint(
    *,
    model: nn.Module,
    path: Path,
    parallel: ParallelContext,
    provenance: dict[str, Any],
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> str | None:
    """Gather a rank-zero CPU checkpoint through the canonical FSDP2 API."""

    reshard_all(model)
    state = get_model_state_dict(model, options=_rank_zero_full_options())
    digest: str | None = None
    if parallel.is_rank_zero:
        if transform is not None:
            state = transform(state)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        torch.save({"model": state, "provenance": provenance}, temporary)
        os.replace(temporary, path)
        digest = sha256_file(path)
        atomic_json_dump(
            {
                "checkpoint": str(path),
                "sha256": digest,
                "provenance": provenance,
            },
            path.with_suffix(path.suffix + ".json"),
        )
    parallel.barrier()
    return digest


def load_portable_payload(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError(f"invalid clean portable checkpoint: {path}")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"checkpoint lacks clean training provenance: {path}")
    if provenance.get("training_roles") != ["train"]:
        raise ValueError(f"checkpoint is not train-only: {path}")
    if provenance.get("test_used") is not False:
        raise ValueError(f"checkpoint provenance does not exclude test use: {path}")
    return payload["model"], provenance
