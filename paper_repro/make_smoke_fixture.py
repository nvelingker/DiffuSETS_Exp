from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from paper_repro.common import atomic_json_dump, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a tiny end-to-end DiffuSETS FSDP2 fixture."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    artifact = root / "artifacts"
    run_root = root / "runs"
    artifact.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    count = args.count
    if count < 16 or count % 4:
        raise ValueError("smoke count must be a multiple of four and at least 16")
    train_count = count // 2
    validation_count = count // 4
    splits = np.asarray(
        ["train"] * train_count
        + ["val"] * validation_count
        + ["test"] * (count - train_count - validation_count)
    )
    subjects = np.arange(10_000, 10_000 + count, dtype=np.int64)
    frame = pd.DataFrame(
        {
            "row_idx": np.arange(count, dtype=np.int64),
            "author_selection_index": np.arange(count, dtype=np.int64),
            "source_manifest_row": np.arange(count, dtype=np.int64),
            "packed_idx": np.arange(count, dtype=np.int64),
            "subject_id": subjects,
            "study_id": np.arange(40_000_000, 40_000_000 + count, dtype=np.int64),
            "waveform_path": [f"toy/{index}" for index in range(count)],
            "split": splits,
            "report": ["Sinus rhythm"] * count,
            "rr_interval_ms": np.full(count, 800.0, dtype=np.float32),
            "sex_male": (np.arange(count) % 2).astype(np.float32),
            "age_years": np.linspace(20, 80, count, dtype=np.float32),
            "conditioning_row": np.arange(count, dtype=np.int64),
        }
    )
    manifest = artifact / "manifest.parquet"
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), manifest)
    waveforms = np.lib.format.open_memmap(
        artifact / "waveforms_float32.npy",
        mode="w+",
        dtype=np.float32,
        shape=(count, 1024, 12),
    )
    waveforms[:] = rng.normal(0, 0.2, size=waveforms.shape).astype(np.float32)
    waveforms.flush()
    del waveforms
    status = np.lib.format.open_memmap(
        artifact / "waveform_status_uint8.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(count,),
    )
    status[:] = 1
    status.flush()
    del status
    heart_rate = np.lib.format.open_memmap(
        artifact / "heart_rate_float32.npy", mode="w+", dtype=np.float32, shape=(count,)
    )
    heart_rate[:] = np.linspace(55, 95, count, dtype=np.float32)
    heart_rate.flush()
    del heart_rate
    source = np.lib.format.open_memmap(
        artifact / "heart_rate_source_uint8.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(count,),
    )
    source[:] = 1
    source.flush()
    del source
    conditioning = np.lib.format.open_memmap(
        artifact / "conditioning_float32.npy",
        mode="w+",
        dtype=np.float32,
        shape=(count, 1539),
    )
    conditioning[:, :1536] = rng.normal(0, 0.02, size=(count, 1536)).astype(np.float32)
    conditioning[:, 1536] = (np.arange(count) % 2).astype(np.float32)
    conditioning[:, 1537] = np.linspace(20, 80, count, dtype=np.float32)
    conditioning[:, 1538] = np.linspace(55, 95, count, dtype=np.float32)
    conditioning.flush()
    del conditioning
    posterior = np.lib.format.open_memmap(
        artifact / "posterior_noise_float32.npy",
        mode="w+",
        dtype=np.float32,
        shape=(count, 4, 128),
    )
    posterior[:] = rng.standard_normal(posterior.shape, dtype=np.float32)
    posterior.flush()
    del posterior
    atomic_json_dump(
        {
            "schema": "diffusets_record_aligned_posterior_noise_v1",
            "seed": 2026,
            "generator": "numpy.PCG64.standard_normal.float32",
            "shape": [count, 4, 128],
            "dtype": "float32",
            "manifest_sha256": sha256_file(manifest),
            "path": str(artifact / "posterior_noise_float32.npy"),
            "sha256": sha256_file(artifact / "posterior_noise_float32.npy"),
        },
        artifact / "posterior_noise_summary.json",
    )
    atomic_json_dump(
        {
            "schema": "diffusets_author_waveform_cache_v1",
            "records": count,
            "complete": count,
            "failed": 0,
            "pending": 0,
        },
        artifact / "waveform_summary.json",
    )

    dummy = artifact / "dummy"
    dummy.write_text("smoke\n")
    config = {
        "schema_version": 1,
        "paths": {
            "ecg_root": str(root),
            "record_list": str(dummy),
            "machine_measurements": str(dummy),
            "shared_manifest": str(dummy),
            "exclude_list": str(dummy),
            "ecgdiff_prepared_manifest": str(dummy),
            "dense_conditioning": str(artifact / "conditioning_float32.npy"),
            "artifact_root": str(artifact),
            "manifest": str(manifest),
            "waveforms": str(artifact / "waveforms_float32.npy"),
            "waveform_status": str(artifact / "waveform_status_uint8.npy"),
            "heart_rate": str(artifact / "heart_rate_float32.npy"),
            "heart_rate_source": str(artifact / "heart_rate_source_uint8.npy"),
            "posterior_noise": str(artifact / "posterior_noise_float32.npy"),
            "latent": str(artifact / "latents_float32.npy"),
            "latent_status": str(artifact / "latent_status_uint8.npy"),
            "run_root": str(run_root),
            "vae_checkpoint": str(run_root / "vae/portable/VAE_model_ep0.pth"),
            "clip_checkpoint": str(run_root / "clip/portable/clip_best.pth"),
            "diffusion_checkpoint": str(run_root / "diffusion/portable/unet_best.pth"),
        },
        "data": {
            "split_seed": 2026,
            "train_fraction": 0.7,
            "validation_fraction": 0.1,
            "source_sample_rate_hz": 500.0,
            "target_samples": 1024,
            "lead_order": [
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
            ],
            "quality_excluded_record_ids": [],
            "posterior_noise_seed": 2026,
            "encode_batch_size_per_rank": 8,
            "expected_counts": {},
            "source_sha256": {},
        },
        "distributed": {
            "backend": "nccl",
            "parameter_dtype": "float32",
            "reduce_dtype": "float32",
            "num_workers_per_rank": 0,
        },
        "vae": {
            "seed": 2026,
            "epochs": 10,
            "global_batch_size": 256,
            "lr": 0.0001,
            "max_lr": 0.0002,
            "save_after_zero_based_epoch": 5,
        },
        "clip": {
            "seed": 2026,
            "embed_dim": 64,
            "epochs": 10,
            "lr": 0.001,
            "weight_decay": 0.001,
            "contrastive_batch_size_per_rank": 256,
            "global_optimizer_batch_size": 16384,
            "validation_batch_size_per_rank": 4,
        },
        "diffusion": {
            "seed": 2026,
            "epochs": 200,
            "global_batch_size": 2048,
            "lr": 0.0001,
            "num_train_steps": 1000,
            "unet_kernel_size": 7,
            "unet_num_levels": 7,
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "initial_best_loss": 50.0,
            "save_every_epochs": 50,
        },
    }
    config_path = root / "smoke_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(config_path)
    print(f"manifest_sha256={sha256_file(manifest)}")


if __name__ == "__main__":
    main()
