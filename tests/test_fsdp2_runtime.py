from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from paper_repro.checkpoint import (
    initialize_adamw_state,
    load_resume_checkpoint,
    save_portable_checkpoint,
    save_resume_checkpoint,
)
from paper_repro.fsdp2 import ParallelContext, apply_fsdp2, require_torch_214
from paper_repro.train import VAETrainingModel


def _init_world_one(tmp_path: Path) -> ParallelContext:
    rendezvous = tmp_path / "rendezvous"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )
    mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("fsdp",))
    return ParallelContext(0, 0, 1, torch.device("cpu"), mesh)


def test_production_vae_plan_is_bottom_up_and_every_parameter_is_dtensor(
    tmp_path: Path,
) -> None:
    require_torch_214()
    parallel = _init_world_one(tmp_path)
    try:
        model, audit = apply_fsdp2(
            VAETrainingModel(),
            stage="vae",
            parallel=parallel,
            parameter_dtype="float32",
        )
        assert audit["application_order"] == "semantic children bottom-up, root last"
        assert audit["fsdp_module_count"] > 2
        assert audit["root_reshard_after_forward"] is False
        assert audit["child_reshard_after_forward"] is True
        assert all(isinstance(parameter, DTensor) for parameter in model.parameters())
        assert all(Path(path).exists() for path in audit["source_files_inspected"])
    finally:
        dist.destroy_process_group()


def test_dcp_resume_and_portable_full_state_round_trip(tmp_path: Path) -> None:
    parallel = _init_world_one(tmp_path)
    try:
        torch.manual_seed(5)
        model = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 2))
        model.register_parameter("unused", nn.Parameter(torch.randn(3)))
        policy = MixedPrecisionPolicy(
            param_dtype=torch.float32, reduce_dtype=torch.float32
        )
        fully_shard(
            model[0], mesh=parallel.mesh, mp_policy=policy, reshard_after_forward=True
        )
        fully_shard(
            model, mesh=parallel.mesh, mp_policy=policy, reshard_after_forward=False
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        initialize_adamw_state(optimizer)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
        x = torch.randn(3, 4)
        loss = model(x).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        expected = get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        provenance = {
            "schema": "diffusets_clean_checkpoint_v1",
            "stage": "tiny",
            "training_roles": ["train"],
            "test_used": False,
            "world_size": 1,
        }
        checkpoint = save_resume_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=tmp_path / "run",
            epoch=1,
            global_step=1,
            parallel=parallel,
            provenance=provenance,
        )
        portable = tmp_path / "portable.pt"
        digest = save_portable_checkpoint(
            model=model,
            path=portable,
            parallel=parallel,
            provenance=provenance,
        )
        assert digest
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.to_local().add_(10)
        epoch, step, loaded_provenance = load_resume_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint=checkpoint,
            parallel=parallel,
        )
        actual = get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        assert epoch == 1 and step == 1
        assert loaded_provenance == provenance
        assert expected.keys() == actual.keys()
        assert all(torch.equal(expected[key], actual[key]) for key in expected)
        payload = torch.load(portable, map_location="cpu", weights_only=False)
        assert payload["model"].keys() == expected.keys()
        assert (
            json.loads(portable.with_suffix(".pt.json").read_text())["sha256"] == digest
        )
    finally:
        dist.destroy_process_group()


def test_adamw_eager_state_keeps_first_update_and_covers_unused_params() -> None:
    torch.manual_seed(19)
    eager_model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
    lazy_model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
    lazy_model.load_state_dict(eager_model.state_dict())
    eager_optimizer = torch.optim.AdamW(eager_model.parameters(), lr=1e-3)
    lazy_optimizer = torch.optim.AdamW(lazy_model.parameters(), lr=1e-3)
    initialize_adamw_state(eager_optimizer)

    x = torch.randn(5, 4)
    eager_model[0](x).square().mean().backward()
    lazy_model[0](x).square().mean().backward()
    eager_optimizer.step()
    lazy_optimizer.step()

    assert len(eager_optimizer.state) == len(tuple(eager_model.parameters()))
    assert len(lazy_optimizer.state) < len(tuple(lazy_model.parameters()))
    assert all(
        torch.equal(eager, lazy)
        for eager, lazy in zip(eager_model.parameters(), lazy_model.parameters())
    )
