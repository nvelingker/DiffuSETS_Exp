from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor


Stage = Literal["vae", "encode", "clip", "diffusion"]


@dataclass(frozen=True, slots=True)
class ParallelContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    mesh: DeviceMesh

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        dist.barrier()

    def mean(self, value: float) -> float:
        tensor = torch.tensor(value, device=self.device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
        return float(tensor.item())

    def sum(self, value: int | float) -> float:
        tensor = torch.tensor(value, device=self.device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor.item())


def require_torch_214() -> None:
    if not torch.__version__.startswith("2.14."):
        raise RuntimeError(
            f"the clean DiffuSETS FSDP2 path requires PyTorch 2.14.x; found {torch.__version__}"
        )
    required = (
        "set_is_last_backward",
        "set_requires_gradient_sync",
        "reshard",
        "unshard",
    )
    missing = [
        name for name in required if not callable(getattr(FSDPModule, name, None))
    ]
    if missing:
        raise RuntimeError(f"installed PyTorch lacks required FSDP2 APIs: {missing}")


def init_parallel(backend: str = "nccl") -> ParallelContext:
    require_torch_214()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be positive")
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL FSDP2 training requires CUDA")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    elif backend == "gloo":
        device = torch.device("cpu")
    else:
        raise ValueError(f"unsupported distributed backend {backend!r}")
    if not dist.is_initialized():
        init_kwargs: dict[str, Any] = {
            "backend": backend,
            "rank": rank,
            "world_size": world_size,
        }
        # PyTorch 2.14 uses device_id to bind NCCL eagerly and to avoid rank to
        # device guessing in collectives such as barrier().
        if backend == "nccl":
            init_kwargs["device_id"] = device
        dist.init_process_group(**init_kwargs)
    mesh = init_device_mesh(device.type, (world_size,), mesh_dim_names=("fsdp",))
    return ParallelContext(rank, local_rank, world_size, device, mesh)


def destroy_parallel() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _mixed_precision(name: str) -> MixedPrecisionPolicy:
    if name == "float32":
        dtype = torch.float32
    elif name == "bfloat16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"unsupported FSDP2 parameter dtype {name!r}")
    return MixedPrecisionPolicy(
        param_dtype=dtype,
        reduce_dtype=torch.float32,
        output_dtype=None,
        cast_forward_inputs=True,
    )


def _fully_shard_once(
    module: nn.Module,
    *,
    mesh: DeviceMesh,
    policy: MixedPrecisionPolicy,
    root: bool = False,
) -> None:
    if isinstance(module, FSDPModule):
        return
    fully_shard(
        module,
        mesh=mesh,
        mp_policy=policy,
        reshard_after_forward=not root,
    )


def _shard_vae_tree(
    model: nn.Module, mesh: DeviceMesh, policy: MixedPrecisionPolicy
) -> None:
    from vae.vae_model import (
        VAE_AttentionBlock,
        VAE_Decoder,
        VAE_Encoder,
        VAE_ResidualBlock,
    )

    for module in tuple(model.modules()):
        if isinstance(module, (VAE_ResidualBlock, VAE_AttentionBlock)):
            _fully_shard_once(module, mesh=mesh, policy=policy)
    for module in tuple(model.modules()):
        if isinstance(module, (VAE_Encoder, VAE_Decoder)) and module is not model:
            _fully_shard_once(module, mesh=mesh, policy=policy)


def _shard_clip_tree(
    model: nn.Module, mesh: DeviceMesh, policy: MixedPrecisionPolicy
) -> None:
    from clip.clip_model import BasicStage, CLIP, Net1D
    from vae.vae_model import VAE_Decoder

    _shard_vae_tree(model, mesh, policy)
    for module in tuple(model.modules()):
        if isinstance(module, BasicStage):
            _fully_shard_once(module, mesh=mesh, policy=policy)
    for module in tuple(model.modules()):
        if isinstance(module, (Net1D, VAE_Decoder)) and module is not model:
            _fully_shard_once(module, mesh=mesh, policy=policy)
    for module in tuple(model.modules()):
        if isinstance(module, CLIP) and module is not model:
            _fully_shard_once(module.text_projector, mesh=mesh, policy=policy)
            _fully_shard_once(module, mesh=mesh, policy=policy)


def _shard_diffusion_tree(
    model: nn.Module, mesh: DeviceMesh, policy: MixedPrecisionPolicy
) -> None:
    from unet.unet_conditional import BottleneckNet, DownsamplingBlock, UpsamplingBlock

    for module in tuple(model.modules()):
        if isinstance(module, (DownsamplingBlock, BottleneckNet, UpsamplingBlock)):
            _fully_shard_once(module, mesh=mesh, policy=policy)
    output = getattr(model, "output_conv", None)
    if isinstance(output, nn.Module):
        _fully_shard_once(output, mesh=mesh, policy=policy)


def apply_fsdp2(
    model: nn.Module,
    *,
    stage: Stage,
    parallel: ParallelContext,
    parameter_dtype: str,
) -> tuple[nn.Module, dict[str, Any]]:
    """Apply FSDP2 bottom-up, following the installed 2.14 implementation."""

    model.to(parallel.device)
    policy = _mixed_precision(parameter_dtype)
    if stage in {"vae", "encode"}:
        _shard_vae_tree(model, parallel.mesh, policy)
    elif stage == "clip":
        _shard_clip_tree(model, parallel.mesh, policy)
    elif stage == "diffusion":
        _shard_diffusion_tree(model, parallel.mesh, policy)
    else:
        raise ValueError(f"unsupported FSDP2 stage {stage!r}")
    _fully_shard_once(model, mesh=parallel.mesh, policy=policy, root=True)

    modules = tuple(
        module for module in model.modules() if isinstance(module, FSDPModule)
    )
    if not isinstance(model, FSDPModule) or len(modules) < 2:
        raise RuntimeError(
            "FSDP2 bottom-up application did not produce child and root groups"
        )
    dtensors = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if isinstance(parameter, DTensor)
    ]
    if len(dtensors) != sum(1 for _ in model.parameters()):
        raise RuntimeError("FSDP2 left one or more parameters unsharded")
    placements = sorted(
        {str(place) for _, parameter in dtensors for place in parameter.placements}
    )
    if not any(value.startswith("S(") or "Shard" in value for value in placements):
        raise RuntimeError(f"FSDP2 parameters are not sharded: placements={placements}")
    audit = {
        "implementation": "torch.distributed.fsdp.fully_shard (FSDP2)",
        "torch_version": torch.__version__,
        "stage": stage,
        "world_size": parallel.world_size,
        "mesh_shape": list(parallel.mesh.shape),
        "mesh_dim_names": list(parallel.mesh.mesh_dim_names or ()),
        "parameter_dtype": parameter_dtype,
        "reduce_dtype": "float32",
        "cast_forward_inputs": True,
        "root_reshard_after_forward": False,
        "child_reshard_after_forward": True,
        "application_order": "semantic children bottom-up, root last",
        "optimizer_construction": "after fully_shard",
        "fsdp_module_count": len(modules),
        "parameter_count": len(dtensors),
        "placements": placements,
        "source_files_inspected": [
            str(
                Path(torch.__file__).parent
                / "distributed/fsdp/_fully_shard/_fully_shard.py"
            ),
            str(
                Path(torch.__file__).parent
                / "distributed/fsdp/_fully_shard/_fsdp_param_group.py"
            ),
            str(
                Path(torch.__file__).parent
                / "distributed/fsdp/_fully_shard/_fsdp_collectives.py"
            ),
        ],
    }
    return model, audit


def set_last_backward(model: nn.Module, is_last_backward: bool) -> None:
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_is_last_backward(is_last_backward)


def set_gradient_sync(model: nn.Module, enabled: bool) -> None:
    """Toggle FSDP2 gradient communication for microbatch accumulation."""

    root_fsdp(model).set_requires_gradient_sync(enabled, recurse=True)


@torch.no_grad()
def average_replicated_float_buffers(model: nn.Module) -> None:
    """Make replicated running-stat buffers identical across parallel ranks."""

    for buffer in model.buffers():
        if buffer.is_floating_point():
            dist.all_reduce(buffer, op=dist.ReduceOp.AVG)
        else:
            minimum = buffer.clone()
            maximum = buffer.clone()
            dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            if not torch.equal(minimum, maximum):
                raise RuntimeError("replicated integral buffers diverged across ranks")


def reshard_all(model: nn.Module) -> None:
    modules = [module for module in model.modules() if isinstance(module, FSDPModule)]
    for module in reversed(modules):
        module.reshard()


def root_fsdp(model: nn.Module) -> FSDPModule:
    if not isinstance(model, FSDPModule):
        raise TypeError("model root is not an FSDP2 module")
    return cast(FSDPModule, model)
