from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paper_repro.common import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class ReproConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def paths(self) -> dict[str, Path]:
        return {name: self.resolve(value) for name, value in self.raw["paths"].items()}

    @property
    def config_sha256(self) -> str:
        return sha256_file(self.path)

    def resolve(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (REPO_ROOT / path).resolve()

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"configuration section {name!r} must be an object")
        return value


def load_config(path: str | Path) -> ReproConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if raw.get("schema_version") != 1:
        raise ValueError(
            "patient-disjoint DiffuSETS config schema_version must equal 1"
        )
    for section in ("paths", "data", "distributed", "vae", "clip", "diffusion"):
        if not isinstance(raw.get(section), dict):
            raise ValueError(f"missing configuration object {section!r}")
    config = ReproConfig(config_path, raw)
    validate_config(config)
    return config


def validate_config(config: ReproConfig) -> None:
    data = config.section("data")
    if int(data.get("split_seed", -1)) != 2026:
        raise ValueError("the shared ECGDiff/SE-Diff split seed is fixed at 2026")
    if float(data.get("train_fraction", -1)) != 0.70:
        raise ValueError("the shared training patient fraction is fixed at 0.70")
    if float(data.get("validation_fraction", -1)) != 0.10:
        raise ValueError("the shared validation patient fraction is fixed at 0.10")
    if int(data.get("target_samples", -1)) != 1024:
        raise ValueError("DiffuSETS' released VAE requires 1,024 samples")
    if list(data.get("lead_order", [])) != [
        "I",
        "II",
        "III",
        "aVR",
        "aVF",
        "aVL",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    ]:
        raise ValueError("lead_order must match the raw MIMIC order used by DiffuSETS")

    vae = config.section("vae")
    required_vae = {
        "epochs": 10,
        "global_batch_size": 256,
        "lr": 1e-4,
        "max_lr": 2e-4,
        "save_after_zero_based_epoch": 5,
    }
    for key, expected in required_vae.items():
        if vae.get(key) != expected:
            raise ValueError(
                f"VAE criterion/default {key!r} is locked to the released value {expected!r}"
            )

    clip = config.section("clip")
    required_clip = {
        "embed_dim": 64,
        "epochs": 10,
        "lr": 1e-3,
        "weight_decay": 1e-3,
        "contrastive_batch_size_per_rank": 256,
        "global_optimizer_batch_size": 16384,
    }
    for key, expected in required_clip.items():
        if clip.get(key) != expected:
            raise ValueError(
                f"DiffuSETS-CLIP64 default {key!r} must equal {expected!r}"
            )

    diffusion = config.section("diffusion")
    diffusion_source = diffusion.get(
        "hyperparameter_source", "released_config_all_legacy"
    )
    shared_diffusion = {
        "epochs": 200,
        "num_train_steps": 1000,
        "unet_kernel_size": 7,
        "unet_num_levels": 7,
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "initial_best_loss": 50.0,
        "save_every_epochs": 50,
    }
    source_specific_diffusion = {
        # The January 2025 release's executable config conflicts with both the
        # paper Methods and the README. Keep it loadable only so the archived
        # seed-2026 v1 run remains auditable.
        "released_config_all_legacy": {
            "global_batch_size": 2048,
            "lr": 1e-4,
        },
        # Patterns 2025 Methods: batch size 512 and learning rate 5e-4. The
        # paper does not state an epoch count, so retain the released code's
        # 200-epoch schedule above.
        "paper_methods": {
            "global_batch_size": 512,
            "lr": 5e-4,
        },
    }
    if diffusion_source not in source_specific_diffusion:
        raise ValueError(
            "diffusion.hyperparameter_source must be one of "
            f"{sorted(source_specific_diffusion)}; found {diffusion_source!r}"
        )
    required_diffusion = {
        **shared_diffusion,
        **source_specific_diffusion[diffusion_source],
    }
    for key, expected in required_diffusion.items():
        if diffusion.get(key) != expected:
            raise ValueError(
                f"DiffuSETS {diffusion_source} setting {key!r} must equal {expected!r}"
            )

    distributed = config.section("distributed")
    if distributed.get("backend") != "nccl":
        raise ValueError("the production FSDP2 path requires the NCCL backend")
    if distributed.get("parameter_dtype") != "float32":
        raise ValueError(
            "the released FP32 training behavior is required; BF16 is incompatible "
            "with DiffuSETS-CLIP64 BatchNorm buffers under FSDP2"
        )
    if distributed.get("reduce_dtype") != "float32":
        raise ValueError("distributed.reduce_dtype must remain float32")


def require_global_batch(global_batch_size: int, world_size: int, *, stage: str) -> int:
    if global_batch_size <= 0:
        raise ValueError(f"{stage} global batch size must be positive")
    if global_batch_size % world_size:
        raise ValueError(
            f"{stage} global batch size {global_batch_size} must be divisible by world size {world_size}"
        )
    return global_batch_size // world_size
