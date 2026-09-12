from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal
import wfdb
from wfdb import processing

from paper_repro.common import atomic_json_dump, sha256_file
from paper_repro.config import ReproConfig, load_config
from paper_repro.data import (
    build_patient_disjoint_manifest,
    load_manifest,
    metadata_heart_rate_from_rr_ms,
    role_counts,
)


_ROWS: pd.DataFrame | None = None
_ECG_ROOT: Path | None = None
_WAVEFORMS: np.memmap | None = None
_STATUS: np.memmap | None = None
_HEART_RATE: np.memmap | None = None
_HR_SOURCE: np.memmap | None = None
_LEAD_ORDER: tuple[str, ...] = ()
_TARGET_SAMPLES = 1024
_SOURCE_SAMPLE_RATE = 500.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare leakage-free DiffuSETS artifacts."
    )
    parser.add_argument(
        "command", choices=("manifest", "waveforms", "posterior-noise", "audit")
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 64))
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--limit", type=int, default=None, help="Development-only row limit."
    )
    return parser.parse_args()


def _create_or_validate_array(
    path: Path,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    overwrite: bool,
    fill: int | float | None = None,
) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and path.exists():
        path.unlink()
    if path.exists():
        array = np.load(path, mmap_mode="r+")
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(
                f"existing {path} has {array.shape}/{array.dtype}, expected {shape}/{dtype}"
            )
        return array
    array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    if fill is not None:
        array[...] = fill
        array.flush()
    return array


def _worker_init(
    waveform_path: str,
    status_path: str,
    heart_rate_path: str,
    source_path: str,
) -> None:
    global _WAVEFORMS, _STATUS, _HEART_RATE, _HR_SOURCE
    _WAVEFORMS = np.load(waveform_path, mmap_mode="r+")
    _STATUS = np.load(status_path, mmap_mode="r+")
    _HEART_RATE = np.load(heart_rate_path, mmap_mode="r+")
    _HR_SOURCE = np.load(source_path, mmap_mode="r+")


def _detect_xqrs_hr(
    raw_signal: np.ndarray[Any, Any], sample_rate: float
) -> float | None:
    for lead in range(raw_signal.shape[1]):
        try:
            detector = processing.XQRS(sig=raw_signal[:, lead], fs=sample_rate)
            detector.detect(verbose=False)
        except Exception:
            continue
        peaks = np.asarray(detector.qrs_inds)
        if len(peaks) > 1:
            rr_seconds = np.diff(peaks) / sample_rate
            if (
                len(rr_seconds)
                and np.isfinite(rr_seconds).all()
                and float(np.mean(rr_seconds)) > 0
            ):
                return 60.0 / float(np.mean(rr_seconds))
    return None


def _prepare_rows(indices: list[int]) -> tuple[int, int, list[dict[str, Any]]]:
    assert _ROWS is not None and _ECG_ROOT is not None
    assert _WAVEFORMS is not None and _STATUS is not None
    assert _HEART_RATE is not None and _HR_SOURCE is not None
    successes = 0
    failures: list[dict[str, Any]] = []
    for row_idx in indices:
        row = _ROWS.iloc[row_idx]
        try:
            values, fields = wfdb.rdsamp(str(_ECG_ROOT / str(row["waveform_path"])))
            values = np.asarray(values)
            sample_rate = float(fields["fs"])
            if not math.isclose(
                sample_rate, _SOURCE_SAMPLE_RATE, rel_tol=0.0, abs_tol=1e-6
            ):
                raise ValueError(
                    f"expected {_SOURCE_SAMPLE_RATE} Hz, found {sample_rate}"
                )
            names = [str(value) for value in fields["sig_name"]]
            if names != list(_LEAD_ORDER):
                raise ValueError(
                    f"expected raw DiffuSETS lead order {list(_LEAD_ORDER)}, found {names}"
                )
            if values.shape != (5000, 12):
                raise ValueError(
                    f"expected waveform shape (5000, 12), found {values.shape}"
                )
            raw_values = values
            values = np.nan_to_num(values)

            heart_rate = metadata_heart_rate_from_rr_ms(row["rr_interval_ms"])
            source = 1
            if heart_rate is None:
                # Match the released fallback input: XQRS sees raw WFDB
                # samples, before the waveform-only NaN replacement.
                heart_rate = _detect_xqrs_hr(raw_values, sample_rate)
                source = 2
            if heart_rate is None or not math.isfinite(heart_rate) or heart_rate <= 0:
                raise ValueError(
                    "RR outside 300--1500 ms and all-lead XQRS fallback failed"
                )

            # This is the released DiffuSETS preprocessing operation, including
            # FFT resampling rather than the polyphase cache used by SE-Diff.
            resampled = signal.resample(values, _TARGET_SAMPLES, axis=0)
            _WAVEFORMS[row_idx] = np.asarray(resampled, dtype=np.float32)
            _HEART_RATE[row_idx] = heart_rate
            _HR_SOURCE[row_idx] = source
            _STATUS[row_idx] = 1
            successes += 1
        except Exception as error:  # noqa: BLE001 - preprocessing must retain row-level failures.
            _STATUS[row_idx] = 2
            _HEART_RATE[row_idx] = np.nan
            _HR_SOURCE[row_idx] = 3
            failures.append(
                {
                    "row_idx": row_idx,
                    "subject_id": int(row["subject_id"]),
                    "study_id": int(row["study_id"]),
                    "error": repr(error),
                }
            )
    return len(indices), successes, failures


def prepare_waveforms(
    config: ReproConfig,
    *,
    workers: int,
    chunksize: int,
    overwrite: bool,
    resume: bool,
    limit: int | None,
) -> dict[str, Any]:
    global _ROWS, _ECG_ROOT, _LEAD_ORDER, _TARGET_SAMPLES, _SOURCE_SAMPLE_RATE
    if workers <= 0 or chunksize <= 0:
        raise ValueError("workers and chunksize must be positive")
    paths = config.paths
    data = config.section("data")
    build_patient_disjoint_manifest(config)
    frame = load_manifest(paths["manifest"])
    n_records = len(frame)
    _ROWS = frame
    _ECG_ROOT = paths["ecg_root"]
    _LEAD_ORDER = tuple(data["lead_order"])
    _TARGET_SAMPLES = int(data["target_samples"])
    _SOURCE_SAMPLE_RATE = float(data["source_sample_rate_hz"])

    waveforms = _create_or_validate_array(
        paths["waveforms"],
        shape=(n_records, _TARGET_SAMPLES, len(_LEAD_ORDER)),
        dtype=np.dtype(np.float32),
        overwrite=overwrite,
    )
    status = _create_or_validate_array(
        paths["waveform_status"],
        shape=(n_records,),
        dtype=np.dtype(np.uint8),
        overwrite=overwrite,
        fill=0,
    )
    heart_rate = _create_or_validate_array(
        paths["heart_rate"],
        shape=(n_records,),
        dtype=np.dtype(np.float32),
        overwrite=overwrite,
        fill=np.nan,
    )
    hr_source = _create_or_validate_array(
        paths["heart_rate_source"],
        shape=(n_records,),
        dtype=np.dtype(np.uint8),
        overwrite=overwrite,
        fill=0,
    )
    del waveforms, heart_rate, hr_source
    if not resume and not overwrite and bool(np.asarray(status).any()):
        raise ValueError("partial waveform cache exists; pass --resume or --overwrite")

    candidates = np.flatnonzero(np.asarray(status) != 1).astype(np.int64)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        candidates = candidates[:limit]
    tasks = [
        candidates[start : start + chunksize].tolist()
        for start in range(0, len(candidates), chunksize)
    ]
    started = time.time()
    failures: list[dict[str, Any]] = []
    processed = 0
    successes = 0
    if tasks:
        context = mp.get_context("fork")
        with context.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(
                str(paths["waveforms"]),
                str(paths["waveform_status"]),
                str(paths["heart_rate"]),
                str(paths["heart_rate_source"]),
            ),
        ) as pool:
            for count, ok, errors in pool.imap_unordered(
                _prepare_rows, tasks, chunksize=1
            ):
                processed += count
                successes += ok
                failures.extend(errors)
                if processed % 10_000 < count:
                    elapsed = max(time.time() - started, 1e-6)
                    print(
                        json.dumps(
                            {
                                "processed_this_run": processed,
                                "pending_at_start": len(candidates),
                                "records_per_second": processed / elapsed,
                                "failures_this_run": len(failures),
                            }
                        ),
                        flush=True,
                    )

    status = np.load(paths["waveform_status"], mmap_mode="r")
    heart_rate = np.load(paths["heart_rate"], mmap_mode="r")
    hr_source = np.load(paths["heart_rate_source"], mmap_mode="r")
    complete = int(np.count_nonzero(status == 1))
    failed = int(np.count_nonzero(status == 2))
    pending = int(np.count_nonzero(status == 0))
    if complete and not np.isfinite(np.asarray(heart_rate)[status == 1]).all():
        raise RuntimeError("completed rows contain non-finite heart rates")
    failure_path = paths["artifact_root"] / "waveform_failures.jsonl"
    if failures:
        with failure_path.open("a", encoding="utf-8") as handle:
            for item in failures:
                handle.write(json.dumps(item, sort_keys=True) + "\n")

    summary: dict[str, Any] = {
        "schema": "diffusets_author_waveform_cache_v1",
        "manifest": str(paths["manifest"]),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "records": n_records,
        "complete": complete,
        "failed": failed,
        "pending": pending,
        "processed_this_run": processed,
        "successful_this_run": successes,
        "elapsed_seconds": time.time() - started,
        "preprocessing": {
            "source_sample_rate_hz": _SOURCE_SAMPLE_RATE,
            "source_samples": 5000,
            "target_samples": _TARGET_SAMPLES,
            "resampler": "scipy.signal.resample_fft",
            "nan_policy": "numpy.nan_to_num",
            "dtype": "float32",
            "lead_order": list(_LEAD_ORDER),
        },
        "heart_rate_policy": {
            "metadata_rr_valid_ms_inclusive": [300.0, 1500.0],
            "metadata_formula": "60000 / rr_interval_ms",
            "fallback": "WFDB XQRS over leads in released order; first lead with >=2 peaks; mean RR",
            "metadata_count": int(np.count_nonzero(hr_source == 1)),
            "xqrs_count": int(np.count_nonzero(hr_source == 2)),
            "failed_count": int(np.count_nonzero(hr_source == 3)),
        },
        "roles_usable": role_counts(frame, status),
        "test_or_validation_used_for_training_selection": False,
    }
    if pending == 0:
        summary["sha256"] = {
            "waveforms": sha256_file(paths["waveforms"]),
            "status": sha256_file(paths["waveform_status"]),
            "heart_rate": sha256_file(paths["heart_rate"]),
            "heart_rate_source": sha256_file(paths["heart_rate_source"]),
        }
    atomic_json_dump(summary, paths["artifact_root"] / "waveform_summary.json")
    return summary


def prepare_posterior_noise(
    config: ReproConfig, *, overwrite: bool = False
) -> dict[str, Any]:
    paths = config.paths
    frame = load_manifest(paths["manifest"])
    output = paths["posterior_noise"]
    summary_path = paths["artifact_root"] / "posterior_noise_summary.json"
    seed = int(config.section("data")["posterior_noise_seed"])
    shape = (len(frame), 4, 128)
    if output.exists() and summary_path.exists() and not overwrite:
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        if summary.get("sha256") != sha256_file(output):
            raise ValueError("posterior-noise cache hash mismatch")
        return summary
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    array = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=shape
    )
    generator = np.random.default_rng(seed)
    chunk = 8192
    for start in range(0, len(frame), chunk):
        stop = min(start + chunk, len(frame))
        array[start:stop] = generator.standard_normal(
            (stop - start, 4, 128), dtype=np.float32
        )
    array.flush()
    del array
    os.replace(temporary, output)
    summary = {
        "schema": "diffusets_record_aligned_posterior_noise_v1",
        "seed": seed,
        "generator": "numpy.PCG64.standard_normal.float32",
        "shape": list(shape),
        "dtype": "float32",
        "manifest_sha256": sha256_file(paths["manifest"]),
        "path": str(output),
        "sha256": sha256_file(output),
    }
    atomic_json_dump(summary, summary_path)
    return summary


def audit_artifacts(config: ReproConfig) -> dict[str, Any]:
    paths = config.paths
    manifest_audit_path = paths["artifact_root"] / "manifest_audit.json"
    waveform_summary_path = paths["artifact_root"] / "waveform_summary.json"
    if not manifest_audit_path.exists() or not waveform_summary_path.exists():
        raise FileNotFoundError(
            "manifest and waveform preparation must complete before audit"
        )
    manifest = load_manifest(paths["manifest"])
    status = np.load(paths["waveform_status"], mmap_mode="r")
    heart_rate = np.load(paths["heart_rate"], mmap_mode="r")
    pending = int(np.count_nonzero(status == 0))
    failed = int(np.count_nonzero(status == 2))
    if pending or failed:
        raise RuntimeError(
            f"waveform cache is not exact: pending={pending}, failed={failed}"
        )
    usable = status == 1
    if not np.isfinite(np.asarray(heart_rate)[usable]).all():
        raise RuntimeError("usable rows contain invalid corrected HR")
    patient_roles = manifest.loc[usable].groupby("subject_id")["split"].nunique()
    if bool(patient_roles.gt(1).any()):
        raise RuntimeError("usable artifacts contain patient contamination")
    actual_roles = role_counts(manifest, status)
    expected_roles = config.section("data").get("expected_counts", {})
    if expected_roles and actual_roles != expected_roles:
        raise RuntimeError(
            f"usable role counts differ from ECGDiff/SE-Diff: {actual_roles}"
        )
    audit = {
        "schema": "diffusets_clean_artifact_audit_v1",
        "manifest_sha256": sha256_file(paths["manifest"]),
        "waveform_status_sha256": sha256_file(paths["waveform_status"]),
        "heart_rate_sha256": sha256_file(paths["heart_rate"]),
        "roles": actual_roles,
        "patient_overlap": {"train_val": 0, "train_test": 0, "val_test": 0},
        "contaminated_released_model_weights_allowed": False,
        "training_roles": ["train"],
        "selection_roles": {"vae": [], "clip": ["val"], "diffusion": ["train"]},
        "final_test_role_used_during_training": False,
    }
    atomic_json_dump(audit, paths["artifact_root"] / "clean_artifact_audit.json")
    return audit


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.command == "manifest":
        result = build_patient_disjoint_manifest(config, overwrite=args.overwrite)
    elif args.command == "waveforms":
        result = prepare_waveforms(
            config,
            workers=args.workers,
            chunksize=args.chunksize,
            overwrite=args.overwrite,
            resume=args.resume,
            limit=args.limit,
        )
    elif args.command == "posterior-noise":
        result = prepare_posterior_noise(config, overwrite=args.overwrite)
    else:
        result = audit_artifacts(config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
