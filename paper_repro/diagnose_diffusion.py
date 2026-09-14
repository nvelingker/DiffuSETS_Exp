from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor

from paper_repro.checkpoint import load_portable_payload
from paper_repro.common import atomic_json_dump, sha256_file
from paper_repro.config import load_config


DEFAULT_TIMESTEPS = (1, 10, 50, 100, 250, 500, 750, 900, 998)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure fixed-noise denoising and reverse-process latent scale for "
            "a clean portable DiffuSETS U-Net checkpoint."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--denoising-batch-size", type=int, default=64)
    parser.add_argument("--sampling-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument(
        "--timesteps",
        default=",".join(str(value) for value in DEFAULT_TIMESTEPS),
        help="Comma-separated training timesteps for the fixed-noise panel.",
    )
    parser.add_argument("--skip-sampling", action="store_true")
    return parser.parse_args(argv)


def _condition_batch(
    *,
    indices: np.ndarray[Any, np.dtype[np.int64]],
    latents: np.ndarray[Any, Any],
    conditioning: np.ndarray[Any, Any],
    heart_rate: np.ndarray[Any, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    latent = torch.from_numpy(
        np.array(latents[indices], dtype=np.float32, copy=True)
    ).to(device)
    values = np.array(conditioning[indices], dtype=np.float32, copy=True)
    text = torch.from_numpy(values[:, :1536]).unsqueeze(1).to(device)
    metadata = {
        "gender": torch.from_numpy(values[:, 1536]).reshape(-1, 1, 1).to(device),
        "age": torch.from_numpy(values[:, 1537]).reshape(-1, 1, 1).to(device),
        "heart rate": torch.from_numpy(
            np.array(heart_rate[indices], dtype=np.float32, copy=True)
        )
        .reshape(-1, 1, 1)
        .to(device),
    }
    return latent, text, metadata


def _even_positions(
    frame: pd.DataFrame, split: str, count: int
) -> np.ndarray[Any, Any]:
    available = np.flatnonzero(frame["split"].astype(str).eq(split).to_numpy())
    if count <= 0 or count > len(available):
        raise ValueError(
            f"invalid {split} diagnostic count {count}; available={len(available)}"
        )
    return available[np.linspace(0, len(available) - 1, count, dtype=np.int64)]


@torch.inference_mode()
def _denoising_panel(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    batch: tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]],
    *,
    timesteps: tuple[int, ...],
    noise_seed: int,
) -> list[dict[str, float | int]]:
    latent, text, metadata = batch
    generator = torch.Generator(device="cpu").manual_seed(noise_seed)
    noise = torch.randn(latent.shape, generator=generator, dtype=torch.float32).to(
        latent.device
    )
    rows: list[dict[str, float | int]] = []
    for value in timesteps:
        timestep = torch.full(
            (len(latent),), value, device=latent.device, dtype=torch.long
        )
        noisy = scheduler.add_noise(latent, noise, timestep)
        prediction = model(noisy, timestep, text, metadata)
        correlation = torch.corrcoef(
            torch.stack((prediction.flatten(), noise.flatten()))
        )[0, 1]
        rows.append(
            {
                "timestep": value,
                "mse_per_element": float(F.mse_loss(prediction, noise)),
                "prediction_std": float(prediction.std()),
                "noise_correlation": float(correlation),
            }
        )
    return rows


@torch.inference_mode()
def _sampling_panel(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    batch: tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]],
    *,
    base_seed: int,
) -> dict[str, Any]:
    target, text, metadata = batch
    count = len(target)
    generators = [
        torch.Generator(device=target.device).manual_seed(base_seed + index)
        for index in range(count)
    ]
    sample = randn_tensor(
        target.shape,
        generator=generators,
        device=target.device,
        dtype=torch.float32,
    )
    trace_iterations = {1, 10, 100, 250, 500, 750, 900, 990, 1000}
    trace: list[dict[str, float | int]] = []
    started = time.monotonic()
    for iteration, timestep in enumerate(scheduler.timesteps, start=1):
        prediction = model(sample, timestep.expand(count), text, metadata)
        sample = scheduler.step(
            model_output=prediction,
            timestep=timestep,
            sample=sample,
            generator=generators,
        ).prev_sample
        if iteration in trace_iterations:
            trace.append(
                {
                    "iteration": iteration,
                    "timestep": int(timestep),
                    "mean": float(sample.mean()),
                    "std": float(sample.std()),
                    "min": float(sample.min()),
                    "max": float(sample.max()),
                    "fraction_abs_ge_0_999": float(
                        (sample.abs() >= 0.999).float().mean()
                    ),
                }
            )
    return {
        "seconds": time.monotonic() - started,
        "target_latent": {
            "mean": float(target.mean()),
            "std": float(target.std()),
            "min": float(target.min()),
            "max": float(target.max()),
        },
        "terminal_latent": trace[-1],
        "trace": trace,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.denoising_batch_size <= 0 or args.sampling_batch_size <= 0:
        raise ValueError("diagnostic batch sizes must be positive")
    timesteps = tuple(int(value) for value in args.timesteps.split(","))
    if not timesteps or any(value < 1 or value > 998 for value in timesteps):
        raise ValueError("diagnostic timesteps must be in [1, 998]")

    config = load_config(args.config)
    paths = config.paths
    checkpoint = args.checkpoint.expanduser().resolve()
    state, provenance = load_portable_payload(checkpoint)
    if provenance.get("stage") != "diffusion":
        raise ValueError("diagnostic checkpoint is not a diffusion U-Net")
    if provenance.get("config_sha256") != config.config_sha256:
        raise ValueError("checkpoint and diagnostic config hashes differ")
    latent_summary_path = paths["artifact_root"] / "latent_summary.json"
    latent_summary = json.loads(latent_summary_path.read_text(encoding="utf-8"))
    if provenance.get("latent_sha256") != latent_summary.get("latent_sha256"):
        raise ValueError("checkpoint and clean latent cache hashes differ")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diagnostic requested but CUDA is unavailable")
    settings = config.section("diffusion")
    from unet.unet_conditional import ECGconditional

    model = ECGconditional(
        int(settings["num_train_steps"]),
        kernel_size=int(settings["unet_kernel_size"]),
        num_levels=int(settings["unet_num_levels"]),
        n_channels=4,
        text_embed_dim=1536,
    )
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval().to(device)

    frame = pd.read_parquet(paths["manifest"])
    latents = np.load(paths["latent"], mmap_mode="r", allow_pickle=False)
    conditioning = np.load(
        paths["dense_conditioning"], mmap_mode="r", allow_pickle=False
    )
    heart_rate = np.load(paths["heart_rate"], mmap_mode="r", allow_pickle=False)
    train = _condition_batch(
        indices=_even_positions(frame, "train", args.denoising_batch_size),
        latents=latents,
        conditioning=conditioning,
        heart_rate=heart_rate,
        device=device,
    )
    test = _condition_batch(
        indices=_even_positions(frame, "test", args.denoising_batch_size),
        latents=latents,
        conditioning=conditioning,
        heart_rate=heart_rate,
        device=device,
    )
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=int(settings["num_train_steps"]),
        beta_start=float(settings["beta_start"]),
        beta_end=float(settings["beta_end"]),
    )
    denoising = {
        "train": _denoising_panel(
            model,
            noise_scheduler,
            train,
            timesteps=timesteps,
            noise_seed=args.seed,
        ),
        "test": _denoising_panel(
            model,
            noise_scheduler,
            test,
            timesteps=timesteps,
            noise_seed=args.seed + 1,
        ),
    }
    sampling: dict[str, Any] | None = None
    if not args.skip_sampling:
        sample_test = _condition_batch(
            indices=_even_positions(frame, "test", args.sampling_batch_size),
            latents=latents,
            conditioning=conditioning,
            heart_rate=heart_rate,
            device=device,
        )
        reverse_scheduler = DDPMScheduler(
            num_train_timesteps=int(settings["num_train_steps"]),
            beta_start=float(settings["beta_start"]),
            beta_end=float(settings["beta_end"]),
        )
        reverse_scheduler.set_timesteps(int(settings["num_train_steps"]), device=device)
        sampling = _sampling_panel(
            model,
            reverse_scheduler,
            sample_test,
            base_seed=args.seed,
        )

    output = {
        "schema": "diffusets_clean_diffusion_checkpoint_diagnostic_v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": provenance.get("epoch"),
        "checkpoint_global_step": provenance.get("global_step"),
        "config": str(config.path),
        "config_sha256": config.config_sha256,
        "manifest_sha256": provenance.get("manifest_sha256"),
        "latent_sha256": provenance.get("latent_sha256"),
        "training_roles": provenance.get("training_roles"),
        "selection_roles": provenance.get("selection_roles"),
        "test_used": provenance.get("test_used"),
        "seed": args.seed,
        "timesteps": list(timesteps),
        "denoising_batch_size": args.denoising_batch_size,
        "sampling_batch_size": None if args.skip_sampling else args.sampling_batch_size,
        "denoising": denoising,
        "sampling": sampling,
    }
    atomic_json_dump(output, args.output.expanduser().resolve())
    print(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
