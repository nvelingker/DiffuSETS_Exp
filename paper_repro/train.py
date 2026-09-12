from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch import nn
from torch.utils.data import DataLoader

from clip.clip_model import CLIP
from paper_repro.checkpoint import (
    find_latest_checkpoint,
    initialize_adamw_state,
    load_portable_payload,
    load_resume_checkpoint,
    save_portable_checkpoint,
    save_resume_checkpoint,
)
from paper_repro.common import atomic_json_dump, seed_everything, sha256_file
from paper_repro.config import ReproConfig, load_config, require_global_batch
from paper_repro.data import (
    DiffuSETSDataset,
    ExactDistributedTrainSampler,
    IndexedDataset,
    PaddedDistributedInferenceSampler,
    load_manifest,
)
from paper_repro.fsdp2 import (
    ParallelContext,
    apply_fsdp2,
    average_replicated_float_buffers,
    destroy_parallel,
    init_parallel,
    set_gradient_sync,
    set_last_backward,
)
from unet.unet_conditional import ECGconditional
from vae.vae_model import VAE_Decoder, VAE_Encoder, loss_function


class VAETrainingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = VAE_Encoder()
        self.decoder = VAE_Decoder()

    def forward(
        self, waveform: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, mean, log_variance = self.encoder(waveform)
        reconstruction = self.decoder(latent)
        return reconstruction, mean, log_variance


class CleanCLIP(CLIP):
    """Released CLIP architecture with an in-forward unscaled-cosine path."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__(embed_dim=embed_dim)
        # PyTorch 2.14 FSDP2 rejects scalar parameters. A length-one parameter
        # broadcasts identically in the released logits calculation.
        self.logit_scale = nn.Parameter(self.logit_scale.detach().reshape(1))

    def load_state_dict(
        self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        # Accept the released scalar representation for compatibility.
        compatible = dict(state_dict)
        if "logit_scale" in compatible:
            compatible["logit_scale"] = compatible["logit_scale"].reshape(1)
        return super().load_state_dict(compatible, strict=strict, assign=assign)

    def forward(
        self,
        signal: torch.Tensor,
        text_embedding: torch.Tensor,
        score_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        signal_embedding = self.encode_signal(signal)
        signal_features = self.ecg_projector(signal_embedding)
        text_features = self.text_projector(text_embedding)
        signal_features = signal_features / signal_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        if score_only:
            return (signal_features * text_features).sum(dim=-1)
        logit_scale = self.logit_scale.exp()
        logits_per_signal = logit_scale * signal_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ signal_features.t()
        return logits_per_signal, logits_per_text


class CLIPTrainingModel(nn.Module):
    def __init__(self, decoder_state: dict[str, Any], embed_dim: int = 64) -> None:
        super().__init__()
        self.decoder = VAE_Decoder()
        self.decoder.load_state_dict(decoder_state, strict=True)
        self.decoder.requires_grad_(False)
        self.decoder.eval()
        self.clip = CleanCLIP(embed_dim=embed_dim)

    def train(self, mode: bool = True) -> CLIPTrainingModel:
        super().train(mode)
        self.decoder.eval()
        return self

    def forward(
        self,
        latent: torch.Tensor,
        text_embedding: torch.Tensor,
        score_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        with torch.no_grad():
            waveform = self.decoder(latent)
        return self.clip(waveform, text_embedding, score_only=score_only)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train clean DiffuSETS stages with PyTorch FSDP2."
    )
    parser.add_argument("stage", choices=("vae", "encode", "clip", "diffusion"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Synchronized smoke/debug stop."
    )
    parser.add_argument(
        "--max-epochs", type=int, default=None, help="Synchronized smoke/debug stop."
    )
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def _git_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ).strip()
        )
    except Exception:  # noqa: BLE001 - provenance should survive exported source trees.
        head, dirty = "unavailable", True
    return {"head": head, "dirty": dirty}


def _logger(stage_dir: Path, parallel: ParallelContext) -> logging.Logger:
    logger = logging.getLogger(f"diffusets-clean-{stage_dir.name}-rank{parallel.rank}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    if parallel.is_rank_zero:
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        logger.addHandler(stream)
        stage_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(stage_dir / "train.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def _loader(
    dataset: DiffuSETSDataset,
    *,
    batch_size: int,
    parallel: ParallelContext,
    seed: int,
    shuffle: bool,
    workers: int,
) -> tuple[DataLoader[dict[str, torch.Tensor]], ExactDistributedTrainSampler]:
    sampler = ExactDistributedTrainSampler(
        len(dataset),
        parallel.rank,
        parallel.world_size,
        shuffle=shuffle,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=parallel.device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=False,
    )
    parallel.require_equal(len(loader), label="training DataLoader step count")
    return loader, sampler


def _inference_loader(
    dataset: DiffuSETSDataset,
    *,
    batch_size: int,
    parallel: ParallelContext,
    workers: int,
) -> DataLoader[dict[str, torch.Tensor]]:
    wrapped = IndexedDataset(dataset)
    sampler = PaddedDistributedInferenceSampler(
        len(dataset), parallel.rank, parallel.world_size
    )
    return DataLoader(
        wrapped,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=parallel.device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=False,
    )


def _base_provenance(
    *,
    stage: str,
    config: ReproConfig,
    parallel: ParallelContext,
    fsdp_audit: dict[str, Any],
    criteria: dict[str, Any],
    selection_roles: list[str],
) -> dict[str, Any]:
    paths = config.paths
    return {
        "schema": "diffusets_clean_checkpoint_v1",
        "stage": stage,
        "created_unix": time.time(),
        "config": str(config.path),
        "config_sha256": config.config_sha256,
        "manifest": str(paths["manifest"]),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "waveform_summary_sha256": sha256_file(
            paths["artifact_root"] / "waveform_summary.json"
        ),
        "training_roles": ["train"],
        "selection_roles": selection_roles,
        "validation_used_for_gradient_updates": False,
        "test_used": False,
        "released_contaminated_weights_loaded": False,
        "criteria": criteria,
        "fsdp2": fsdp_audit,
        "world_size": parallel.world_size,
        "torch_version": torch.__version__,
        "git": _git_state(),
    }


def _validate_dependency_provenance(
    provenance: dict[str, Any], *, config: ReproConfig, expected_stage: str
) -> None:
    if provenance.get("stage") != expected_stage:
        raise ValueError(
            f"expected clean {expected_stage} checkpoint, found {provenance.get('stage')!r}"
        )
    current_manifest = sha256_file(config.paths["manifest"])
    if provenance.get("manifest_sha256") != current_manifest:
        raise ValueError("checkpoint and current patient-disjoint manifest differ")
    if provenance.get("config_sha256") != config.config_sha256:
        raise ValueError("checkpoint and current clean-run configuration differ")
    if (
        provenance.get("training_roles") != ["train"]
        or provenance.get("test_used") is not False
    ):
        raise ValueError("dependency checkpoint does not prove train-only exposure")


def _load_clean_latent_summary(config: ReproConfig) -> dict[str, Any]:
    paths = config.paths
    summary_path = paths["artifact_root"] / "latent_summary.json"
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    required = {
        "manifest_sha256": sha256_file(paths["manifest"]),
        "latent_sha256": sha256_file(paths["latent"]),
        "status_sha256": sha256_file(paths["latent_status"]),
    }
    mismatches = {
        key: {"expected": expected, "found": summary.get(key)}
        for key, expected in required.items()
        if summary.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"latent-cache provenance mismatch: {mismatches}")
    if summary.get("pending") != 0 or summary.get("excluded_waveform_failures") != 0:
        raise ValueError("latent cache is incomplete or excludes waveform failures")
    if summary.get("records") != len(load_manifest(paths["manifest"])):
        raise ValueError("latent cache record count differs from the clean manifest")
    if summary.get("vae_checkpoint_sha256") != sha256_file(paths["vae_checkpoint"]):
        raise ValueError("latent cache was produced by a different VAE checkpoint")
    if summary.get("test_or_validation_used_to_fit_encoder") is not False:
        raise ValueError("latent cache does not prove a train-only fitted encoder")
    return summary


def _save_fsdp_audit(
    stage_dir: Path, fsdp_audit: dict[str, Any], parallel: ParallelContext
) -> None:
    if parallel.is_rank_zero:
        atomic_json_dump(fsdp_audit, stage_dir / "fsdp2_audit.json")
    parallel.barrier()


def _vae_portable_transform(state: dict[str, Any]) -> dict[str, Any]:
    encoder = {
        key.removeprefix("encoder."): value
        for key, value in state.items()
        if key.startswith("encoder.")
    }
    decoder = {
        key.removeprefix("decoder."): value
        for key, value in state.items()
        if key.startswith("decoder.")
    }
    if not encoder or not decoder or len(encoder) + len(decoder) != len(state):
        raise RuntimeError(
            "VAE portable state did not partition into encoder and decoder"
        )
    return {"encoder": encoder, "decoder": decoder}


def _clip_portable_transform(state: dict[str, Any]) -> dict[str, Any]:
    result = {
        key.removeprefix("clip."): value
        for key, value in state.items()
        if key.startswith("clip.")
    }
    if not result:
        raise RuntimeError("CLIP portable state contains no CLIP parameters")
    # Preserve the released checkpoint representation even though FSDP2 needs
    # this parameter to be length one while training.
    result["logit_scale"] = result["logit_scale"].reshape(())
    return result


def _stage_epochs(configured: int, maximum: int | None) -> int:
    if maximum is None:
        return configured
    if maximum <= 0:
        raise ValueError("max-epochs must be positive")
    return min(configured, maximum)


def _final_batch_rank_scale(
    local_examples: int, *, is_final: bool, parallel: ParallelContext
) -> float:
    """Correct FSDP's rank average for an uneven, unpadded final batch."""

    if not is_final:
        return 1.0
    global_examples = parallel.sum(local_examples)
    if global_examples <= 0:
        raise ValueError("final distributed batch is empty")
    return parallel.world_size * local_examples / global_examples


def train_vae(
    config: ReproConfig,
    parallel: ParallelContext,
    *,
    resume: bool,
    max_steps: int | None,
    max_epochs: int | None,
    workers_override: int | None,
) -> None:
    paths = config.paths
    settings = config.section("vae")
    distributed = config.section("distributed")
    stage_dir = paths["run_root"] / "vae"
    logger = _logger(stage_dir, parallel)
    workers = int(
        distributed["num_workers_per_rank"]
        if workers_override is None
        else workers_override
    )
    dataset = DiffuSETSDataset(
        manifest_path=paths["manifest"],
        split="train",
        purpose="train_vae",
        waveform_path=paths["waveforms"],
        status_path=paths["waveform_status"],
    )
    local_batch = require_global_batch(
        int(settings["global_batch_size"]), parallel.world_size, stage="VAE"
    )
    loader, sampler = _loader(
        dataset,
        batch_size=local_batch,
        parallel=parallel,
        seed=int(settings["seed"]),
        shuffle=False,
        workers=workers,
    )

    seed_everything(int(settings["seed"]))
    model: nn.Module = VAETrainingModel()
    model, fsdp_audit = apply_fsdp2(
        model,
        stage="vae",
        parallel=parallel,
        parameter_dtype=str(distributed["parameter_dtype"]),
    )
    _save_fsdp_audit(stage_dir, fsdp_audit, parallel)
    seed_everything(int(settings["seed"]) + parallel.rank + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["lr"]))
    initialize_adamw_state(optimizer)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(settings["max_lr"]),
        epochs=int(settings["epochs"]),
        steps_per_epoch=len(loader),
    )
    criteria = {
        "objective": "released vae.loss_function without modification",
        "reconstruction": "sum MSE divided by batch size",
        "kl": "Normal KL; sum latent channel then mean",
        "kl_epoch_weight": "(zero_based_epoch + 1) / 10",
        "optimizer": "AdamW",
        "lr": float(settings["lr"]),
        "one_cycle_max_lr": float(settings["max_lr"]),
        "epochs": int(settings["epochs"]),
        "global_batch_size": int(settings["global_batch_size"]),
        "record_order": "manifest order; released VAE loader used shuffle=False",
        "checkpoint_rule": "save zero-based epochs > 5; prescribed final is ep9",
    }
    provenance = _base_provenance(
        stage="vae",
        config=config,
        parallel=parallel,
        fsdp_audit=fsdp_audit,
        criteria=criteria,
        selection_roles=[],
    )
    start_epoch = 1
    global_step = 0
    if resume:
        latest = find_latest_checkpoint(stage_dir)
        if latest is not None:
            completed_epoch, global_step, prior = load_resume_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint=latest,
                parallel=parallel,
            )
            _validate_dependency_provenance(prior, config=config, expected_stage="vae")
            start_epoch = completed_epoch + 1
            logger.info("resumed %s at global step %d", latest, global_step)

    configured_epochs = int(settings["epochs"])
    stop_epoch = _stage_epochs(configured_epochs, max_epochs)
    interrupted = False
    for epoch in range(start_epoch, stop_epoch + 1):
        sampler.set_epoch(epoch)
        model.train()
        weighted_loss = weighted_mse = weighted_kl = examples = 0.0
        kld_weight = epoch / configured_epochs
        for batch_index, batch in enumerate(loader):
            waveform = batch["waveform"].to(parallel.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            set_last_backward(model, True)
            reconstruction, mean, log_variance = model(waveform)
            losses = loss_function(
                reconstruction, waveform, mean, log_variance, kld_weight
            )
            gradient_scale = _final_batch_rank_scale(
                len(waveform),
                is_final=batch_index == len(loader) - 1,
                parallel=parallel,
            )
            (losses["loss"] * gradient_scale).backward()
            optimizer.step()
            scheduler.step()
            count = float(len(waveform))
            weighted_loss += float(losses["loss"].detach()) * count
            weighted_mse += float(losses["mse"]) * count
            weighted_kl += float(losses["KLD"]) * count
            examples += count
            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                interrupted = True
                break
        global_examples = parallel.sum(examples)
        metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "loss": parallel.sum(weighted_loss) / global_examples,
            "mse": parallel.sum(weighted_mse) / global_examples,
            "kld": parallel.sum(weighted_kl) / global_examples,
            "kld_weight": kld_weight,
            "lr": scheduler.get_last_lr()[0],
        }
        logger.info("epoch metrics %s", json.dumps(metrics, sort_keys=True))
        epoch_provenance = {
            **provenance,
            "epoch": epoch,
            "global_step": global_step,
            "metrics": metrics,
        }
        save_resume_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=stage_dir,
            epoch=epoch,
            global_step=global_step,
            parallel=parallel,
            provenance=epoch_provenance,
        )
        if epoch - 1 > int(settings["save_after_zero_based_epoch"]) or interrupted:
            compatibility_name = f"VAE_model_ep{epoch - 1}.pth"
            portable_path = stage_dir / "portable" / compatibility_name
            save_portable_checkpoint(
                model=model,
                path=portable_path,
                parallel=parallel,
                provenance=epoch_provenance,
                transform=_vae_portable_transform,
            )
            if epoch == configured_epochs:
                configured_path = paths["vae_checkpoint"]
                if portable_path != configured_path:
                    save_portable_checkpoint(
                        model=model,
                        path=configured_path,
                        parallel=parallel,
                        provenance=epoch_provenance,
                        transform=_vae_portable_transform,
                    )
        if interrupted:
            break


def encode_latents(
    config: ReproConfig,
    parallel: ParallelContext,
    *,
    max_steps: int | None,
    workers_override: int | None,
) -> None:
    paths = config.paths
    settings = config.section("data")
    distributed = config.section("distributed")
    stage_dir = paths["run_root"] / "encode"
    logger = _logger(stage_dir, parallel)
    workers = int(
        distributed["num_workers_per_rank"]
        if workers_override is None
        else workers_override
    )
    state, vae_provenance = load_portable_payload(paths["vae_checkpoint"])
    _validate_dependency_provenance(vae_provenance, config=config, expected_stage="vae")
    if not isinstance(state.get("encoder"), dict):
        raise ValueError("clean VAE checkpoint lacks encoder state")
    seed_everything(int(settings["posterior_noise_seed"]))
    encoder: nn.Module = VAE_Encoder()
    encoder.load_state_dict(state["encoder"], strict=True)
    encoder.requires_grad_(False)
    encoder.eval()
    encoder, fsdp_audit = apply_fsdp2(
        encoder,
        stage="encode",
        parallel=parallel,
        parameter_dtype=str(distributed["parameter_dtype"]),
    )
    encoder.eval()
    _save_fsdp_audit(stage_dir, fsdp_audit, parallel)

    dataset = DiffuSETSDataset(
        manifest_path=paths["manifest"],
        split="all",
        purpose="encode",
        waveform_path=paths["waveforms"],
        status_path=paths["waveform_status"],
    )
    batch_size = int(settings["encode_batch_size_per_rank"])
    loader = _inference_loader(
        dataset,
        batch_size=batch_size,
        parallel=parallel,
        workers=workers,
    )
    frame = load_manifest(paths["manifest"])
    if parallel.is_rank_zero:
        paths["latent"].parent.mkdir(parents=True, exist_ok=True)
        if paths["latent"].exists():
            latent = np.load(paths["latent"], mmap_mode="r")
            if latent.shape != (len(frame), 4, 128) or latent.dtype != np.float32:
                raise ValueError("existing latent cache has the wrong shape or dtype")
            del latent
        else:
            latent = np.lib.format.open_memmap(
                paths["latent"], mode="w+", dtype=np.float32, shape=(len(frame), 4, 128)
            )
            latent.flush()
            del latent
        if paths["latent_status"].exists():
            status = np.load(paths["latent_status"], mmap_mode="r")
            if status.shape != (len(frame),) or status.dtype != np.uint8:
                raise ValueError("existing latent status has the wrong shape or dtype")
            del status
        else:
            status = np.lib.format.open_memmap(
                paths["latent_status"], mode="w+", dtype=np.uint8, shape=(len(frame),)
            )
            waveform_status = np.load(paths["waveform_status"], mmap_mode="r")
            status[:] = np.where(waveform_status == 1, 0, 2).astype(np.uint8)
            status.flush()
            del status
    parallel.barrier()
    latent_out = np.load(paths["latent"], mmap_mode="r+")
    latent_status = np.load(paths["latent_status"], mmap_mode="r+")
    posterior_noise = np.load(paths["posterior_noise"], mmap_mode="r")
    if (
        posterior_noise.shape != (len(frame), 4, 128)
        or posterior_noise.dtype != np.float32
    ):
        raise ValueError("posterior-noise cache is not manifest-aligned float32")
    posterior_summary_path = paths["artifact_root"] / "posterior_noise_summary.json"
    with posterior_summary_path.open(encoding="utf-8") as handle:
        posterior_summary = json.load(handle)
    if posterior_summary.get("manifest_sha256") != sha256_file(
        paths["manifest"]
    ) or posterior_summary.get("sha256") != sha256_file(paths["posterior_noise"]):
        raise ValueError("posterior-noise cache provenance mismatch")

    processed = 0
    steps = 0
    started = time.time()
    with torch.no_grad():
        for batch in loader:
            row_idx = batch["row_idx"].numpy().astype(np.int64)
            valid = batch["valid"].numpy().astype(bool)
            waveform = batch["waveform"].to(parallel.device, non_blocking=True)
            noise = torch.from_numpy(np.array(posterior_noise[row_idx], copy=True)).to(
                parallel.device, non_blocking=True
            )
            encoded, _, _ = encoder(waveform, noise)
            encoded_cpu = encoded.float().cpu().numpy()
            for local_index in np.flatnonzero(valid):
                destination = int(row_idx[local_index])
                latent_out[destination] = encoded_cpu[local_index]
                latent_status[destination] = 1
            processed += int(valid.sum())
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
    latent_out.flush()
    latent_status.flush()
    del latent_out, latent_status
    parallel.barrier()
    if parallel.is_rank_zero:
        status = np.load(paths["latent_status"], mmap_mode="r")
        waveform_status = np.load(paths["waveform_status"], mmap_mode="r")
        expected = waveform_status == 1
        complete = int(np.count_nonzero(status == 1))
        pending = int(np.count_nonzero(expected & (status != 1)))
        summary: dict[str, Any] = {
            "schema": "diffusets_clean_latent_cache_v1",
            "manifest_sha256": sha256_file(paths["manifest"]),
            "vae_checkpoint": str(paths["vae_checkpoint"]),
            "vae_checkpoint_sha256": sha256_file(paths["vae_checkpoint"]),
            "vae_training_roles": vae_provenance["training_roles"],
            "posterior_noise": str(paths["posterior_noise"]),
            "posterior_noise_sha256": sha256_file(paths["posterior_noise"]),
            "records": len(frame),
            "complete": complete,
            "pending": pending,
            "excluded_waveform_failures": int(np.count_nonzero(~expected)),
            "shape": [len(frame), 4, 128],
            "dtype": "float32",
            "elapsed_seconds": time.time() - started,
            "test_or_validation_used_to_fit_encoder": False,
        }
        if pending == 0:
            summary["latent_sha256"] = sha256_file(paths["latent"])
            summary["status_sha256"] = sha256_file(paths["latent_status"])
        atomic_json_dump(summary, paths["artifact_root"] / "latent_summary.json")
        logger.info("latent summary %s", json.dumps(summary, sort_keys=True))
    parallel.barrier()


def _buffered_batches(
    loader: DataLoader[dict[str, torch.Tensor]], accumulation_steps: int
) -> Any:
    group: list[dict[str, torch.Tensor]] = []
    for batch in loader:
        group.append(batch)
        if len(group) == accumulation_steps:
            yield group
            group = []
    if group:
        yield group


@torch.no_grad()
def evaluate_clip(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    parallel: ParallelContext,
) -> float:
    model.eval()
    local_sum = 0.0
    local_count = 0
    for batch in loader:
        latent = batch["latent"].to(parallel.device, non_blocking=True)
        text = batch["text_embedding"].to(parallel.device, non_blocking=True)
        scores = model(latent, text, True)
        if not isinstance(scores, torch.Tensor):
            raise TypeError("CLIP score-only forward returned logits")
        valid = batch["valid"].to(parallel.device)
        local_sum += float(scores[valid].float().sum())
        local_count += int(valid.sum())
    total = parallel.sum(local_sum)
    count = parallel.sum(local_count)
    return total / count


def train_clip(
    config: ReproConfig,
    parallel: ParallelContext,
    *,
    resume: bool,
    max_steps: int | None,
    max_epochs: int | None,
    workers_override: int | None,
) -> None:
    paths = config.paths
    settings = config.section("clip")
    distributed = config.section("distributed")
    stage_dir = paths["run_root"] / "clip"
    logger = _logger(stage_dir, parallel)
    workers = int(
        distributed["num_workers_per_rank"]
        if workers_override is None
        else workers_override
    )
    vae_state, vae_provenance = load_portable_payload(paths["vae_checkpoint"])
    _validate_dependency_provenance(vae_provenance, config=config, expected_stage="vae")
    if not isinstance(vae_state.get("decoder"), dict):
        raise ValueError("clean VAE checkpoint lacks decoder state")
    _load_clean_latent_summary(config)
    train_dataset = DiffuSETSDataset(
        manifest_path=paths["manifest"],
        split="train",
        purpose="train_clip",
        latent_path=paths["latent"],
        conditioning_path=paths["dense_conditioning"],
        status_path=paths["latent_status"],
    )
    validation_dataset = DiffuSETSDataset(
        manifest_path=paths["manifest"],
        split="val",
        purpose="validate_clip",
        latent_path=paths["latent"],
        conditioning_path=paths["dense_conditioning"],
        status_path=paths["latent_status"],
    )
    local_microbatch = int(settings["contrastive_batch_size_per_rank"])
    global_optimizer_batch = int(settings["global_optimizer_batch_size"])
    global_microbatch = local_microbatch * parallel.world_size
    if global_optimizer_batch % global_microbatch:
        raise ValueError(
            "CLIP global optimizer batch must be divisible by rank-local contrastive batches"
        )
    accumulation_steps = global_optimizer_batch // global_microbatch
    train_loader, train_sampler = _loader(
        train_dataset,
        batch_size=local_microbatch,
        parallel=parallel,
        seed=int(settings["seed"]),
        shuffle=True,
        workers=workers,
    )
    validation_loader = _inference_loader(
        validation_dataset,
        batch_size=int(settings["validation_batch_size_per_rank"]),
        parallel=parallel,
        workers=workers,
    )

    seed_everything(int(settings["seed"]))
    model: nn.Module = CLIPTrainingModel(
        vae_state["decoder"], embed_dim=int(settings["embed_dim"])
    )
    model, fsdp_audit = apply_fsdp2(
        model,
        stage="clip",
        parallel=parallel,
        parameter_dtype=str(distributed["parameter_dtype"]),
    )
    _save_fsdp_audit(stage_dir, fsdp_audit, parallel)
    seed_everything(int(settings["seed"]) + parallel.rank + 1)
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(settings["lr"]),
        weight_decay=float(settings["weight_decay"]),
    )
    initialize_adamw_state(optimizer)
    criteria = {
        "architecture": "released DiffuSETS CLIP64",
        "objective": "symmetric ECG/text cross entropy",
        "contrastive_batch_size_per_rank": local_microbatch,
        "global_optimizer_batch_size": global_optimizer_batch,
        "record_order": "seeded shuffle; released CLIP loader used shuffle=True",
        "optimizer": "AdamW",
        "lr": float(settings["lr"]),
        "weight_decay": float(settings["weight_decay"]),
        "epochs": int(settings["epochs"]),
        "checkpoint_selection": "maximum cosine on patient-disjoint validation",
        "decoder": "frozen train-only VAE",
        "batchnorm_running_stats": "rank-average before validation/checkpoint",
    }
    provenance = _base_provenance(
        stage="clip",
        config=config,
        parallel=parallel,
        fsdp_audit=fsdp_audit,
        criteria=criteria,
        selection_roles=["val"],
    )
    provenance["vae_checkpoint_sha256"] = sha256_file(paths["vae_checkpoint"])
    start_epoch = 1
    global_step = 0
    best_score = -math.inf
    if resume:
        latest = find_latest_checkpoint(stage_dir)
        if latest is not None:
            completed_epoch, global_step, prior = load_resume_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=None,
                checkpoint=latest,
                parallel=parallel,
            )
            _validate_dependency_provenance(prior, config=config, expected_stage="clip")
            start_epoch = completed_epoch + 1
            best_score = float(prior.get("best_validation_clip", -math.inf))

    stop_epoch = _stage_epochs(int(settings["epochs"]), max_epochs)
    interrupted = False
    criterion = nn.CrossEntropyLoss()
    for epoch in range(start_epoch, stop_epoch + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        local_weighted_loss = 0.0
        local_examples = 0
        optimizer_groups = math.ceil(len(train_loader) / accumulation_steps)
        for group_index, group in enumerate(
            _buffered_batches(train_loader, accumulation_steps)
        ):
            optimizer.zero_grad(set_to_none=True)
            group_examples = sum(len(batch["latent"]) for batch in group)
            rank_scale = _final_batch_rank_scale(
                group_examples,
                is_final=group_index == optimizer_groups - 1,
                parallel=parallel,
            )
            for index, batch in enumerate(group):
                latent = batch["latent"].to(parallel.device, non_blocking=True)
                text = batch["text_embedding"].to(parallel.device, non_blocking=True)
                is_last_microbatch = index == len(group) - 1
                set_gradient_sync(model, is_last_microbatch)
                set_last_backward(model, is_last_microbatch)
                output = model(latent, text, False)
                if not isinstance(output, tuple):
                    raise TypeError(
                        "CLIP training forward returned scores instead of logits"
                    )
                logits_signal, logits_text = output
                labels = torch.arange(len(latent), device=parallel.device)
                loss = (
                    criterion(logits_signal, labels) + criterion(logits_text, labels)
                ) / 2
                weight = len(latent) / group_examples
                (loss * weight * rank_scale).backward()
                local_weighted_loss += float(loss.detach()) * len(latent)
                local_examples += len(latent)
            optimizer.step()
            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                interrupted = True
                break
        average_replicated_float_buffers(model)
        validation_score = evaluate_clip(model, validation_loader, parallel)
        global_examples = parallel.sum(local_examples)
        metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": parallel.sum(local_weighted_loss) / global_examples,
            "validation_clip": validation_score,
        }
        logger.info("epoch metrics %s", json.dumps(metrics, sort_keys=True))
        improved = validation_score > best_score
        best_score = max(best_score, validation_score)
        epoch_provenance = {
            **provenance,
            "epoch": epoch,
            "global_step": global_step,
            "metrics": metrics,
            "best_validation_clip": best_score,
        }
        save_resume_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            output_dir=stage_dir,
            epoch=epoch,
            global_step=global_step,
            parallel=parallel,
            provenance=epoch_provenance,
        )
        save_portable_checkpoint(
            model=model,
            path=stage_dir / "portable" / f"clip_model_ep{epoch}.pth",
            parallel=parallel,
            provenance=epoch_provenance,
            transform=_clip_portable_transform,
        )
        if improved:
            save_portable_checkpoint(
                model=model,
                path=paths["clip_checkpoint"],
                parallel=parallel,
                provenance=epoch_provenance,
                transform=_clip_portable_transform,
            )
        if interrupted:
            break


def _condition_tensors(
    batch: dict[str, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    text = batch["text_embedding"].to(device, non_blocking=True).unsqueeze(1)
    condition = {
        "gender": batch["gender"].to(device, non_blocking=True).reshape(-1, 1, 1),
        "age": batch["age"].to(device, non_blocking=True).reshape(-1, 1, 1),
        "heart rate": batch["heart_rate"]
        .to(device, non_blocking=True)
        .reshape(-1, 1, 1),
    }
    return text, condition


def train_diffusion(
    config: ReproConfig,
    parallel: ParallelContext,
    *,
    resume: bool,
    max_steps: int | None,
    max_epochs: int | None,
    workers_override: int | None,
) -> None:
    paths = config.paths
    settings = config.section("diffusion")
    distributed = config.section("distributed")
    stage_dir = paths["run_root"] / "diffusion"
    logger = _logger(stage_dir, parallel)
    workers = int(
        distributed["num_workers_per_rank"]
        if workers_override is None
        else workers_override
    )
    latent_summary = _load_clean_latent_summary(config)
    dataset = DiffuSETSDataset(
        manifest_path=paths["manifest"],
        split="train",
        purpose="train_diffusion",
        latent_path=paths["latent"],
        conditioning_path=paths["dense_conditioning"],
        heart_rate_path=paths["heart_rate"],
        status_path=paths["latent_status"],
    )
    local_batch = require_global_batch(
        int(settings["global_batch_size"]), parallel.world_size, stage="diffusion"
    )
    loader, sampler = _loader(
        dataset,
        batch_size=local_batch,
        parallel=parallel,
        seed=int(settings["seed"]),
        shuffle=False,
        workers=workers,
    )
    seed_everything(int(settings["seed"]))
    model: nn.Module = ECGconditional(
        int(settings["num_train_steps"]),
        kernel_size=int(settings["unet_kernel_size"]),
        num_levels=int(settings["unet_num_levels"]),
        n_channels=4,
        text_embed_dim=1536,
    )
    model, fsdp_audit = apply_fsdp2(
        model,
        stage="diffusion",
        parallel=parallel,
        parameter_dtype=str(distributed["parameter_dtype"]),
    )
    _save_fsdp_audit(stage_dir, fsdp_audit, parallel)
    seed_everything(int(settings["seed"]) + parallel.rank + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["lr"]))
    initialize_adamw_state(optimizer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(settings["epochs"]) * len(loader),
        eta_min=0.1 * float(settings["lr"]),
    )
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=int(settings["num_train_steps"]),
        beta_start=float(settings["beta_start"]),
        beta_end=float(settings["beta_end"]),
    )
    criteria = {
        "objective": "released noise-prediction sum MSE divided by batch size",
        "timesteps": "uniform integers [1, num_train_steps - 2]",
        "scheduler": "Diffusers DDPM linear beta schedule",
        "optimizer": "AdamW",
        "lr": float(settings["lr"]),
        "lr_scheduler": "CosineAnnealingLR eta_min=0.1*lr",
        "epochs": int(settings["epochs"]),
        "global_batch_size": int(settings["global_batch_size"]),
        "record_order": "manifest order; released diffusion loader used shuffle=False",
        "checkpoint_selection": "minimum training epoch loss, as released",
    }
    provenance = _base_provenance(
        stage="diffusion",
        config=config,
        parallel=parallel,
        fsdp_audit=fsdp_audit,
        criteria=criteria,
        selection_roles=["train"],
    )
    provenance["latent_sha256"] = latent_summary.get("latent_sha256")
    start_epoch = 1
    global_step = 0
    best_loss = float(settings["initial_best_loss"])
    if resume:
        latest = find_latest_checkpoint(stage_dir)
        if latest is not None:
            completed_epoch, global_step, prior = load_resume_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint=latest,
                parallel=parallel,
            )
            _validate_dependency_provenance(
                prior, config=config, expected_stage="diffusion"
            )
            start_epoch = completed_epoch + 1
            best_loss = float(prior.get("best_training_loss", best_loss))

    stop_epoch = _stage_epochs(int(settings["epochs"]), max_epochs)
    interrupted = False
    for epoch in range(start_epoch, stop_epoch + 1):
        sampler.set_epoch(epoch)
        model.train()
        weighted_loss = examples = 0.0
        for batch_index, batch in enumerate(loader):
            latent = batch["latent"].to(parallel.device, non_blocking=True)
            text, condition = _condition_tensors(batch, parallel.device)
            timesteps = torch.randint(
                1,
                int(settings["num_train_steps"]) - 1,
                (len(latent),),
                device=parallel.device,
            )
            noise = torch.randn_like(latent)
            noisy = noise_scheduler.add_noise(latent, noise, timesteps)
            optimizer.zero_grad(set_to_none=True)
            set_last_backward(model, True)
            prediction = model(noisy, timesteps, text, condition)
            loss = F.mse_loss(prediction.float(), noise.float(), reduction="sum").div(
                len(noise)
            )
            gradient_scale = _final_batch_rank_scale(
                len(latent),
                is_final=batch_index == len(loader) - 1,
                parallel=parallel,
            )
            (loss * gradient_scale).backward()
            optimizer.step()
            scheduler.step()
            weighted_loss += float(loss.detach()) * len(latent)
            examples += len(latent)
            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                interrupted = True
                break
        global_examples = parallel.sum(examples)
        epoch_loss = parallel.sum(weighted_loss) / global_examples
        metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "training_loss": epoch_loss,
            "lr": scheduler.get_last_lr()[0],
        }
        logger.info("epoch metrics %s", json.dumps(metrics, sort_keys=True))
        improved = epoch_loss < best_loss
        best_loss = min(best_loss, epoch_loss)
        epoch_provenance = {
            **provenance,
            "epoch": epoch,
            "global_step": global_step,
            "metrics": metrics,
            "best_training_loss": best_loss,
        }
        save_resume_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=stage_dir,
            epoch=epoch,
            global_step=global_step,
            parallel=parallel,
            provenance=epoch_provenance,
        )
        if improved:
            save_portable_checkpoint(
                model=model,
                path=paths["diffusion_checkpoint"],
                parallel=parallel,
                provenance=epoch_provenance,
            )
        if epoch % int(settings["save_every_epochs"]) == 0 or interrupted:
            save_portable_checkpoint(
                model=model,
                path=stage_dir / "portable" / f"unet_{epoch}.pth",
                parallel=parallel,
                provenance=epoch_provenance,
            )
        if interrupted:
            break


def main() -> None:
    args = parse_args()
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max-steps must be positive")
    config = load_config(args.config)
    parallel = init_parallel(str(config.section("distributed")["backend"]))
    try:
        if args.stage == "vae":
            train_vae(
                config,
                parallel,
                resume=args.resume,
                max_steps=args.max_steps,
                max_epochs=args.max_epochs,
                workers_override=args.num_workers,
            )
        elif args.stage == "encode":
            encode_latents(
                config,
                parallel,
                max_steps=args.max_steps,
                workers_override=args.num_workers,
            )
        elif args.stage == "clip":
            train_clip(
                config,
                parallel,
                resume=args.resume,
                max_steps=args.max_steps,
                max_epochs=args.max_epochs,
                workers_override=args.num_workers,
            )
        else:
            train_diffusion(
                config,
                parallel,
                resume=args.resume,
                max_steps=args.max_steps,
                max_epochs=args.max_epochs,
                workers_override=args.num_workers,
            )
    finally:
        destroy_parallel()


if __name__ == "__main__":
    main()
