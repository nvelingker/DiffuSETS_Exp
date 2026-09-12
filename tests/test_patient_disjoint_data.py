from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataset.mimic_iv_ecg_dataset import heart_rate_from_rr_interval
from paper_repro.common import patient_split
from paper_repro.config import load_config
from paper_repro.data import (
    DiffuSETSDataset,
    PaddedDistributedInferenceSampler,
    load_manifest,
    metadata_heart_rate_from_rr_ms,
    resolve_heart_rate,
)


def test_rr_metadata_uses_millisecond_bounds_and_formula() -> None:
    assert metadata_heart_rate_from_rr_ms(300) == pytest.approx(200.0)
    assert metadata_heart_rate_from_rr_ms(800) == pytest.approx(75.0)
    assert metadata_heart_rate_from_rr_ms(1500) == pytest.approx(40.0)
    assert metadata_heart_rate_from_rr_ms(299) is None
    assert metadata_heart_rate_from_rr_ms(1501) is None
    assert metadata_heart_rate_from_rr_ms(float("nan")) is None


def test_rr_fallback_only_runs_outside_valid_metadata_range() -> None:
    calls = 0

    def detector() -> float:
        nonlocal calls
        calls += 1
        return 88.0

    value, source = resolve_heart_rate(600, detector)
    assert value == pytest.approx(100.0)
    assert source == "metadata_rr_interval"
    assert calls == 0
    value, source = resolve_heart_rate(0, detector)
    assert value == pytest.approx(88.0)
    assert source == "wfdb_xqrs"
    assert calls == 1
    assert heart_rate_from_rr_interval(1000, detector) == pytest.approx(60.0)
    assert calls == 1
    assert heart_rate_from_rr_interval(65535, detector) == pytest.approx(88.0)
    assert calls == 2


def test_rr_fallback_failure_is_explicit() -> None:
    with pytest.raises(ValueError, match="XQRS"):
        resolve_heart_rate(0, lambda: None)
    with pytest.raises(ValueError, match="XQRS"):
        heart_rate_from_rr_interval(0, lambda: None)


def _write_manifest(path: Path, splits: list[str], subjects: list[int]) -> None:
    count = len(splits)
    frame = pd.DataFrame(
        {
            "row_idx": np.arange(count, dtype=np.int64),
            "author_selection_index": np.arange(count, dtype=np.int64),
            "source_manifest_row": np.arange(count, dtype=np.int64),
            "packed_idx": np.arange(count, dtype=np.int64),
            "subject_id": subjects,
            "study_id": np.arange(100, 100 + count),
            "waveform_path": [f"record/{i}" for i in range(count)],
            "split": splits,
            "report": ["sinus rhythm"] * count,
            "rr_interval_ms": [800.0] * count,
            "sex_male": [0.0] * count,
            "age_years": [50.0] * count,
            "conditioning_row": np.arange(count, dtype=np.int64),
        }
    )
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)


def test_manifest_rejects_patient_crossing_roles(tmp_path: Path) -> None:
    path = tmp_path / "manifest.parquet"
    _write_manifest(path, ["train", "test"], [7, 7])
    with pytest.raises(ValueError, match="multiple roles"):
        load_manifest(path)


def test_training_dataset_refuses_nontrain_rows(tmp_path: Path) -> None:
    path = tmp_path / "manifest.parquet"
    _write_manifest(path, ["train", "val", "test"], [1, 2, 3])
    waveforms = np.lib.format.open_memmap(
        tmp_path / "waveforms.npy", mode="w+", dtype=np.float32, shape=(3, 1024, 12)
    )
    waveforms[:] = 0
    waveforms.flush()
    status = np.lib.format.open_memmap(
        tmp_path / "status.npy", mode="w+", dtype=np.uint8, shape=(3,)
    )
    status[:] = 1
    status.flush()
    with pytest.raises(ValueError, match="only train"):
        DiffuSETSDataset(
            manifest_path=path,
            split="val",
            purpose="train_vae",
            waveform_path=tmp_path / "waveforms.npy",
            status_path=tmp_path / "status.npy",
        )


def test_training_dataset_refuses_incomplete_cache_rows(tmp_path: Path) -> None:
    path = tmp_path / "manifest.parquet"
    _write_manifest(path, ["train", "train"], [1, 2])
    waveforms = np.lib.format.open_memmap(
        tmp_path / "waveforms.npy", mode="w+", dtype=np.float32, shape=(2, 1024, 12)
    )
    waveforms[:] = 0
    waveforms.flush()
    status = np.lib.format.open_memmap(
        tmp_path / "status.npy", mode="w+", dtype=np.uint8, shape=(2,)
    )
    status[:] = [1, 2]
    status.flush()
    with pytest.raises(ValueError, match="complete clean cache"):
        DiffuSETSDataset(
            manifest_path=path,
            split="train",
            purpose="train_vae",
            waveform_path=tmp_path / "waveforms.npy",
            status_path=tmp_path / "status.npy",
        )


def test_patient_hash_is_stable_and_role_is_subject_only() -> None:
    values = [patient_split(subject) for subject in (10000032, 10000117, 19999987)]
    assert values == [
        patient_split(subject) for subject in (10000032, 10000117, 19999987)
    ]
    assert all(value in {"train", "val", "test"} for value in values)


def test_inference_sampler_marks_empty_ranks_as_padding() -> None:
    assert list(PaddedDistributedInferenceSampler(2, rank=0, world_size=4)) == [
        (0, True)
    ]
    assert list(PaddedDistributedInferenceSampler(2, rank=2, world_size=4)) == [
        (0, False)
    ]


def test_production_config_locks_released_vae_criteria(tmp_path: Path) -> None:
    production = (
        Path(__file__).resolve().parents[1] / "config/patient_disjoint_fsdp2.json"
    )
    config = load_config(production)
    assert config.section("vae") == {
        "seed": 2026,
        "epochs": 10,
        "global_batch_size": 256,
        "lr": 1e-4,
        "max_lr": 2e-4,
        "save_after_zero_based_epoch": 5,
    }
    raw = json.loads(production.read_text())
    raw["vae"]["epochs"] = 11
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="VAE criterion"):
        load_config(invalid)

    raw = json.loads(production.read_text())
    raw["clip"]["embed_dim"] = 128
    invalid.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="CLIP64"):
        load_config(invalid)

    raw = json.loads(production.read_text())
    raw["distributed"]["parameter_dtype"] = "bfloat16"
    invalid.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="released FP32"):
        load_config(invalid)
