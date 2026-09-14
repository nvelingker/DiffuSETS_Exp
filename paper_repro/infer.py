from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.distributed as dist
from diffusers import DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor

from paper_repro.config import load_config
from paper_repro.evaluation import (
    ACTIVE_CONFIG_SHA256,
    ACTIVE_SUITE_ID,
    ACTIVE_UNET_SHA256,
    ACTIVE_VAE_SHA256,
    CANONICAL_LEADS,
    DIFFUSETS_LEADS,
    SCHEMA,
    active_suite_provenance,
    atomic_json_dump,
    canonicalize_decoded,
    condition_identity_sha256,
    load_mimic_conditions,
    load_ptbxl_conditions,
    repository_state,
    save_array,
    sha256_file,
    validate_active_checkpoint,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config/patient_disjoint_fsdp2.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a hash-bound evaluation panel with the clean patient-disjoint "
            "DiffuSETS seed-2026 U-Net and VAE."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--condition-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--condition-source",
        choices=("clean-mimic", "ptbxl-author-package"),
        required=True,
    )
    parser.add_argument("--ptbxl-package", type=Path, default=REPO_ROOT / "prerequisites/ptbxl_vae.pt")
    parser.add_argument("--text-embeddings", type=Path, default=None)
    parser.add_argument("--text-embeddings-sha256", default=None)
    parser.add_argument("--base-seed", type=int, default=20260822)
    parser.add_argument("--batch-size-per-rank", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--reuse-complete", action="store_true")
    parser.add_argument(
        "--allow-quarantined-v1-audit",
        action="store_true",
        help=(
            "Permit reproduction of the known-undertrained v1 output for an "
            "explicit diagnostic audit. Never use this for a new result."
        ),
    )
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="Development-only escape hatch. Registered production outputs require clean Git source.",
    )
    return parser.parse_args(argv)


def _load_models(
    config: Any, device: torch.device
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    from unet.unet_conditional import ECGconditional
    from vae.vae_model import VAE_Decoder

    paths = config.paths
    suite = active_suite_provenance(config)
    vae_state, _ = validate_active_checkpoint(
        paths["vae_checkpoint"],
        expected_sha256=ACTIVE_VAE_SHA256,
        expected_stage="vae",
        config=config,
    )
    unet_state, _ = validate_active_checkpoint(
        paths["diffusion_checkpoint"],
        expected_sha256=ACTIVE_UNET_SHA256,
        expected_stage="diffusion",
        config=config,
    )
    decoder_state = vae_state.get("decoder")
    if not isinstance(decoder_state, dict):
        raise ValueError("active clean VAE checkpoint lacks decoder state")
    diffusion = config.section("diffusion")
    denoiser = ECGconditional(
        int(diffusion["num_train_steps"]),
        kernel_size=int(diffusion["unet_kernel_size"]),
        num_levels=int(diffusion["unet_num_levels"]),
        n_channels=4,
        text_embed_dim=1536,
    )
    denoiser.load_state_dict(unet_state, strict=True)
    decoder = VAE_Decoder()
    decoder.load_state_dict(decoder_state, strict=True)
    return (
        denoiser.requires_grad_(False).eval().to(device),
        decoder.requires_grad_(False).eval().to(device),
        suite,
    )


def make_scheduler(
    *, num_train_steps: int, beta_start: float, beta_end: float, inference_steps: int, device: torch.device
) -> Any:
    scheduler: Any = cast(Any, DDPMScheduler)(
        num_train_timesteps=num_train_steps,
        beta_start=beta_start,
        beta_end=beta_end,
    )
    scheduler.set_timesteps(inference_steps, device=device)
    return scheduler


@torch.inference_mode()
def sample_latents(
    denoiser: torch.nn.Module,
    *,
    text: torch.Tensor,
    metadata: dict[str, torch.Tensor],
    seeds: tuple[int, ...],
    scheduler: Any,
    device: torch.device,
    progress: Any = None,
) -> torch.Tensor:
    count = len(seeds)
    if text.shape != (count, 1, 1536):
        raise ValueError("DiffuSETS text input must be [batch,1,1536]")
    if set(metadata) != {"gender", "age", "heart rate"} or any(
        value.shape != (count, 1, 1) for value in metadata.values()
    ):
        raise ValueError("DiffuSETS scalar inputs must each be [batch,1,1]")
    generators = [
        torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds
    ]
    sample = randn_tensor(
        (count, 4, 128),
        generator=generators,
        device=device,
        dtype=torch.float32,
    )
    total_steps = len(scheduler.timesteps)
    for step_index, timestep in enumerate(scheduler.timesteps, start=1):
        predicted_noise = denoiser(
            sample,
            timestep.expand(count),
            text,
            metadata,
        )
        sample = scheduler.step(
            model_output=predicted_noise,
            timestep=timestep,
            sample=sample,
            generator=generators,
        ).prev_sample
        if progress is not None:
            progress(step_index, total_steps)
    if sample.shape != (count, 4, 128) or not bool(torch.isfinite(sample).all()):
        raise FloatingPointError("clean DiffuSETS reverse process returned malformed latents")
    return sample


@torch.inference_mode()
def generate_local(
    *,
    denoiser: torch.nn.Module,
    decoder: torch.nn.Module,
    text: np.ndarray,
    metadata: np.ndarray,
    global_start: int,
    base_seed: int,
    batch_size: int,
    scheduler_settings: dict[str, float | int],
    device: torch.device,
    progress_every: int,
    progress: Any,
) -> torch.Tensor:
    generated: list[torch.Tensor] = []
    count = len(text)
    for local_start in range(0, count, batch_size):
        local_end = min(local_start + batch_size, count)
        global_positions = range(global_start + local_start, global_start + local_end)
        seeds = tuple(base_seed + position for position in global_positions)
        text_tensor = torch.from_numpy(
            np.array(text[local_start:local_end], dtype=np.float32, copy=True)
        ).to(device)[:, None]
        scalar = torch.from_numpy(
            np.array(metadata[local_start:local_end], dtype=np.float32, copy=True)
        ).to(device)
        condition = {
            "gender": scalar[:, 0].reshape(-1, 1, 1),
            "age": scalar[:, 1].reshape(-1, 1, 1),
            "heart rate": scalar[:, 2].reshape(-1, 1, 1),
        }
        scheduler = make_scheduler(device=device, **scheduler_settings)

        def batch_progress(step: int, steps: int) -> None:
            if step == 1 or step % progress_every == 0 or step == steps:
                progress(local_start, local_end, step, steps)

        latent = sample_latents(
            denoiser,
            text=text_tensor,
            metadata=condition,
            seeds=seeds,
            scheduler=scheduler,
            device=device,
            progress=batch_progress,
        )
        decoded = canonicalize_decoded(decoder(latent))
        if decoded.shape != (local_end - local_start, 12, 1024):
            raise ValueError("clean DiffuSETS decoder returned the wrong waveform shape")
        if not bool(torch.isfinite(decoded).all()):
            raise FloatingPointError("clean DiffuSETS decoder returned nonfinite waveforms")
        generated.append(decoded.float())
    return (
        torch.cat(generated, dim=0)
        if generated
        else torch.empty((0, 12, 1024), device=device, dtype=torch.float32)
    )


def _validate_reusable_output(
    output: Path,
    *,
    condition_summary_sha256: str,
    records: int,
    base_seed: int,
) -> dict[str, Any]:
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        raise FileExistsError(f"output exists without a complete summary: {output}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required = {
        "schema": SCHEMA,
        "stage": "ddpm_cfg_inference",
        "status": "complete",
        "records": records,
        "base_seed": base_seed,
        "condition_summary_sha256": condition_summary_sha256,
        "suite_id": ACTIVE_SUITE_ID,
        "config_sha256": ACTIVE_CONFIG_SHA256,
        "diffusion_checkpoint_sha256": ACTIVE_UNET_SHA256,
        "vae_checkpoint_sha256": ACTIVE_VAE_SHA256,
    }
    for key, expected in required.items():
        if summary.get(key) != expected:
            raise ValueError(f"reusable inference field {key!r} changed")
    for key in (
        "waveforms",
        "seeds",
        "status_array",
        "conditioning",
        "source_rows",
    ):
        path = Path(summary[key])
        if not path.is_file() or sha256_file(path) != summary.get(f"{key}_sha256"):
            raise ValueError(f"reusable inference artifact {key!r} changed")
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.allow_quarantined_v1_audit:
        raise RuntimeError(
            "the registered seed-2026 v1 diffusion U-Net is quarantined: it used "
            "the conflicting released JSON batch/lr instead of the paper Methods. "
            "Pass --allow-quarantined-v1-audit only to reproduce the archived failure."
        )
    if args.base_seed < 0 or args.batch_size_per_rank <= 0 or args.progress_every <= 0:
        raise ValueError("seed and batching/progress values must be positive")
    if args.condition_source == "ptbxl-author-package" and (
        args.text_embeddings is None or args.text_embeddings_sha256 is None
    ):
        raise ValueError("PTB-XL inference requires explicit text embeddings and SHA-256")
    config = load_config(args.config)
    if config.config_sha256 != ACTIVE_CONFIG_SHA256:
        raise ValueError("inference config is not the registered clean-suite config")
    root_state = repository_state(REPO_ROOT)
    if root_state["dirty"] and not args.allow_dirty_source:
        raise RuntimeError("production inference requires a clean DiffuSETS Git checkout")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("clean DiffuSETS evaluation inference requires CUDA")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid distributed rank environment")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(hours=2), device_id=device)
    output = args.output_dir.expanduser().resolve()
    started = time.monotonic()
    try:
        conditions = (
            load_mimic_conditions(config, args.condition_dir)
            if args.condition_source == "clean-mimic"
            else load_ptbxl_conditions(
                args.condition_dir,
                text_path=args.text_embeddings,
                text_sha256=args.text_embeddings_sha256,
                package_path=args.ptbxl_package,
            )
        )
        count = len(conditions.frame)
        reuse = False
        existing: dict[str, Any] | None = None
        if rank == 0:
            if output.exists():
                if not args.reuse_complete:
                    raise FileExistsError(output)
                existing = _validate_reusable_output(
                    output,
                    condition_summary_sha256=conditions.summary_sha256,
                    records=count,
                    base_seed=args.base_seed,
                )
                reuse = True
            else:
                output.mkdir(parents=True)
                atomic_json_dump(
                    {
                        "status": "generating",
                        "records": count,
                        "world_size": world_size,
                        "started_utc": datetime.now(UTC).isoformat(),
                    },
                    output / "status.json",
                )
        reuse_tensor = torch.tensor([int(reuse)], device=device, dtype=torch.uint8)
        dist.broadcast(reuse_tensor, src=0)
        if int(reuse_tensor.item()):
            if rank == 0:
                print(json.dumps(existing, indent=2, sort_keys=True), flush=True)
            return

        denoiser, decoder, suite = _load_models(config, device)
        per_rank = math.ceil(count / world_size)
        start = min(rank * per_rank, count)
        end = min(start + per_rank, count)
        diffusion = config.section("diffusion")
        scheduler_settings: dict[str, float | int] = {
            "num_train_steps": int(diffusion["num_train_steps"]),
            "beta_start": float(diffusion["beta_start"]),
            "beta_end": float(diffusion["beta_end"]),
            "inference_steps": 1000,
        }

        def write_progress(
            local_start: int, local_end: int, step: int, step_count: int
        ) -> None:
            if rank != 0:
                return
            payload = {
                "stage": "denoising",
                "rank_zero_global_rows": [start + local_start, start + local_end],
                "denoising_step": step,
                "denoising_steps": step_count,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "timestamp_utc": datetime.now(UTC).isoformat(),
            }
            with (output / "progress.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, allow_nan=False, sort_keys=True) + "\n")
            atomic_json_dump(payload, output / "progress.json")

        local = generate_local(
            denoiser=denoiser,
            decoder=decoder,
            text=conditions.text[start:end],
            metadata=conditions.metadata[start:end],
            global_start=start,
            base_seed=args.base_seed,
            batch_size=args.batch_size_per_rank,
            scheduler_settings=scheduler_settings,
            device=device,
            progress_every=args.progress_every,
            progress=write_progress,
        )
        padded = torch.zeros((per_rank, 12, 1024), dtype=torch.float32, device=device)
        padded[: end - start] = local
        gathered = [torch.empty_like(padded) for _ in range(world_size)] if rank == 0 else None
        dist.gather(padded, gather_list=gathered, dst=0)
        if rank == 0:
            assert gathered is not None
            pieces: list[torch.Tensor] = []
            for owner, values in enumerate(gathered):
                owner_start = min(owner * per_rank, count)
                owner_end = min(owner_start + per_rank, count)
                pieces.append(values[: owner_end - owner_start].cpu())
            combined = torch.cat(pieces).numpy()[None]
            if combined.shape != (1, count, 12, 1024) or not np.isfinite(combined).all():
                raise RuntimeError("gathered clean DiffuSETS panel is malformed")
            seeds = np.arange(
                args.base_seed, args.base_seed + count, dtype=np.int64
            )[None]
            status = np.ones((1, count), dtype=np.uint8)
            paths = {
                "waveforms": output / "waveforms_float32.npy",
                "seeds": output / "seeds_int64.npy",
                "status_array": output / "status_uint8.npy",
                "conditioning": output / "conditioning_float32.npy",
                "source_rows": output / "source_rows_int64.npy",
            }
            hashes = {
                name: save_array(path, values)
                for name, path, values in (
                    ("waveforms", paths["waveforms"], combined.astype(np.float32, copy=False)),
                    ("seeds", paths["seeds"], seeds),
                    ("status_array", paths["status_array"], status),
                    ("conditioning", paths["conditioning"], conditions.metadata),
                    ("source_rows", paths["source_rows"], conditions.source_rows),
                )
            }
            elapsed = time.monotonic() - started
            inference = {
                "schema": SCHEMA,
                "stage": "ddpm_cfg_inference",
                "status": "complete",
                "dataset": conditions.summary["dataset"],
                "records": count,
                "samples_per_condition": 1,
                "waveform_shape": [12, 1024],
                "waveform_dtype": "float32",
                "condition_dir": str(args.condition_dir.expanduser().resolve()),
                "condition_summary_sha256": conditions.summary_sha256,
                "condition_identity_sha256": condition_identity_sha256(conditions),
                "condition_source": args.condition_source,
                "conditioning_provenance": conditions.provenance,
                "text_embeddings": str(conditions.text_path),
                "text_embeddings_sha256": conditions.text_sha256,
                "waveforms": str(paths["waveforms"]),
                "waveforms_sha256": hashes["waveforms"],
                "seeds": str(paths["seeds"]),
                "seeds_sha256": hashes["seeds"],
                "status_array": str(paths["status_array"]),
                "status_array_sha256": hashes["status_array"],
                "status_sha256": hashes["status_array"],
                "conditioning": str(paths["conditioning"]),
                "conditioning_sha256": hashes["conditioning"],
                "conditioning_columns": ["gender_male", "age_years", "heart_rate_bpm"],
                "source_rows": str(paths["source_rows"]),
                "source_rows_sha256": hashes["source_rows"],
                "suite_id": ACTIVE_SUITE_ID,
                "config": str(config.path),
                "config_sha256": ACTIVE_CONFIG_SHA256,
                "diffusion_checkpoint": suite["diffusion_checkpoint"],
                "diffusion_checkpoint_sha256": ACTIVE_UNET_SHA256,
                "checkpoint_epoch": suite["training_epoch"],
                "checkpoint_step": suite["training_global_step"],
                "vae_checkpoint": suite["vae_checkpoint"],
                "vae_checkpoint_sha256": ACTIVE_VAE_SHA256,
                "sampler": "ancestral_DDPM",
                "guidance_scale": 1.0,
                "prediction_type": "epsilon",
                "inference_steps": 1000,
                "timesteps": 1000,
                "beta_schedule": "linear",
                "beta_start": scheduler_settings["beta_start"],
                "beta_end": scheduler_settings["beta_end"],
                "variance_type": "fixed_small",
                "clip_predicted_x0": True,
                "base_seed": args.base_seed,
                "seed_policy": "base_seed + zero_based_condition_position",
                "source_lead_order": list(DIFFUSETS_LEADS),
                "output_lead_order": list(CANONICAL_LEADS),
                "lead_adapter": "swap decoder positions 4/5 (aVF/aVL) into canonical aVL/aVF order",
                "world_size": world_size,
                "rank_partition": "contiguous_ceil",
                "batch_size_per_rank": args.batch_size_per_rank,
                "hardware_disclosure": f"independent model replica inference across {world_size} GPUs",
                "torch_version": torch.__version__,
                "repository": str(REPO_ROOT),
                "repository_state": root_state,
                "source_files_sha256": {
                    "infer.py": sha256_file(Path(__file__)),
                    "evaluation.py": sha256_file(Path(__file__).with_name("evaluation.py")),
                },
                "elapsed_seconds": elapsed,
                "completed_utc": datetime.now(UTC).isoformat(),
            }
            atomic_json_dump(inference, output / "summary.json")
            atomic_json_dump(
                {
                    "status": "complete",
                    "summary": str((output / "summary.json").resolve()),
                    "summary_sha256": sha256_file(output / "summary.json"),
                    "elapsed_seconds": elapsed,
                },
                output / "status.json",
            )
            print(json.dumps(inference, indent=2, sort_keys=True), flush=True)
        dist.barrier(device_ids=[local_rank])
    except BaseException as error:
        if rank == 0 and output.is_dir():
            atomic_json_dump(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                },
                output / "status.json",
            )
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
