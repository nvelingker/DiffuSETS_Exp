from __future__ import annotations

import ast
import json
import math
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler

from paper_repro.common import atomic_json_dump, patient_split, sha256_file, sha256_json
from paper_repro.config import ReproConfig


Role = Literal["train", "val", "test", "all"]
TRAINING_PURPOSES = frozenset({"train_vae", "train_clip", "train_diffusion"})
REPORT_COLUMNS = tuple(f"report_{index}" for index in range(18))


def metadata_heart_rate_from_rr_ms(rr_interval_ms: Any) -> float | None:
    """Interpret MIMIC machine-measurement RR in milliseconds.

    The released code first divided by 1,000 and then compared the resulting
    seconds against millisecond bounds. The intended 300--1,500 ms gate is
    applied before conversion here.
    """

    try:
        rr_ms = float(rr_interval_ms)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(rr_ms) or not 300.0 <= rr_ms <= 1500.0:
        return None
    return 60_000.0 / rr_ms


def resolve_heart_rate(
    rr_interval_ms: Any,
    detector: Callable[[], float | None],
) -> tuple[float, str]:
    metadata_hr = metadata_heart_rate_from_rr_ms(rr_interval_ms)
    if metadata_hr is not None:
        return metadata_hr, "metadata_rr_interval"
    detected = detector()
    if detected is None or not math.isfinite(float(detected)) or float(detected) <= 0:
        raise ValueError("invalid RR interval and XQRS could not estimate heart rate")
    return float(detected), "wfdb_xqrs"


def _read_exclusion_indices(path: Path) -> list[int]:
    values: list[int] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            value = ast.literal_eval(text)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid exclusion index at {path}:{line_number}")
            values.append(value)
    if len(values) != len(set(values)):
        raise ValueError("DiffuSETS exclusion indices contain duplicates")
    return values


def _joined_reports(row: pd.Series) -> str:
    values = [str(row[name]) for name in REPORT_COLUMNS if isinstance(row[name], str)]
    return "|".join(values)


def _require_hash(path: Path, expected: str | None, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if expected and actual != expected:
        raise ValueError(f"{label} hash mismatch: expected {expected}, found {actual}")
    return actual


def build_patient_disjoint_manifest(
    config: ReproConfig, *, overwrite: bool = False
) -> dict[str, Any]:
    """Reconstruct author selection and bind it to the shared patient roles."""

    paths = config.paths
    data = config.section("data")
    output_dir = paths["artifact_root"]
    output_path = paths["manifest"]
    audit_path = output_dir / "manifest_audit.json"
    if output_path.exists() and audit_path.exists() and not overwrite:
        with audit_path.open(encoding="utf-8") as handle:
            audit = json.load(handle)
        if audit.get("manifest_sha256") != sha256_file(output_path):
            raise ValueError(
                "existing manifest does not match its audit; pass --overwrite to rebuild"
            )
        return audit

    hashes = data.get("source_sha256", {})
    source_hashes = {
        "record_list": _require_hash(
            paths["record_list"], hashes.get("record_list"), "record list"
        ),
        "machine_measurements": _require_hash(
            paths["machine_measurements"],
            hashes.get("machine_measurements"),
            "machine measurements",
        ),
        "shared_manifest": _require_hash(
            paths["shared_manifest"],
            hashes.get("shared_manifest"),
            "shared patient manifest",
        ),
        "exclude_list": _require_hash(
            paths["exclude_list"],
            hashes.get("exclude_list"),
            "DiffuSETS exclusion list",
        ),
        "ecgdiff_prepared_manifest": _require_hash(
            paths["ecgdiff_prepared_manifest"],
            hashes.get("ecgdiff_prepared_manifest"),
            "ECGDiff prepared manifest",
        ),
        "dense_conditioning": _require_hash(
            paths["dense_conditioning"],
            hashes.get("dense_conditioning"),
            "released conditioning cache",
        ),
    }

    records = pd.read_csv(
        paths["record_list"],
        usecols=["subject_id", "study_id", "path"],
        low_memory=False,
    )
    measurements = pd.read_csv(
        paths["machine_measurements"],
        usecols=["subject_id", "study_id", "rr_interval", *REPORT_COLUMNS],
        low_memory=False,
    )
    selected = records.merge(
        measurements, how="inner", on=["subject_id", "study_id"], sort=False
    )
    source_merged_records = len(selected)
    exclusions = _read_exclusion_indices(paths["exclude_list"])
    missing_exclusion_indices = sorted(set(exclusions).difference(selected.index))
    if missing_exclusion_indices:
        raise ValueError(
            f"exclusion indices are outside the ordered merge: {missing_exclusion_indices[:10]}"
        )
    selected = selected.drop(index=exclusions)

    quality_excluded_ids = {
        int(value) for value in data.get("quality_excluded_record_ids", [])
    }
    quality_mask = selected["study_id"].astype(np.int64).isin(quality_excluded_ids)
    found_quality_ids = set(selected.loc[quality_mask, "study_id"].astype(int))
    if found_quality_ids != quality_excluded_ids:
        missing = sorted(quality_excluded_ids.difference(found_quality_ids))
        raise ValueError(f"configured quality exclusions were not present: {missing}")
    selected = selected.loc[~quality_mask].copy()
    selected["report"] = selected.loc[:, REPORT_COLUMNS].apply(_joined_reports, axis=1)

    shared = pd.read_csv(
        paths["shared_manifest"],
        usecols=[
            "manifest_row",
            "subject_id",
            "study_id",
            "waveform_path",
            "split",
            "packed_idx",
        ],
        low_memory=False,
    )
    if shared["study_id"].duplicated().any():
        raise ValueError("shared patient manifest contains duplicate study IDs")
    selected = selected.merge(
        shared,
        on=["subject_id", "study_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_shared"),
        sort=False,
    )
    if selected[["manifest_row", "split", "packed_idx"]].isna().any().any():
        raise ValueError(
            "author-selected records are missing from the shared patient manifest"
        )

    split_seed = int(data["split_seed"])
    train_fraction = float(data["train_fraction"])
    validation_fraction = float(data["validation_fraction"])
    expected_split = selected["subject_id"].map(
        lambda subject: patient_split(
            int(subject),
            seed=split_seed,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
    )
    split_mismatches = int((expected_split != selected["split"].astype(str)).sum())
    if split_mismatches:
        raise ValueError(
            f"shared manifest has {split_mismatches} deterministic patient-split mismatches"
        )

    prepared = pq.read_table(
        paths["ecgdiff_prepared_manifest"], columns=["record_id", "group_id", "split"]
    ).to_pandas()
    expected_records = selected["study_id"].astype(str).to_numpy()
    if len(prepared) != len(selected) or not np.array_equal(
        prepared["record_id"].astype(str).to_numpy(), expected_records
    ):
        raise ValueError(
            "reconstructed DiffuSETS selection is not row-identical to ECGDiff"
        )
    if not np.array_equal(
        prepared["group_id"].astype(str).to_numpy(),
        selected["subject_id"].astype(str).to_numpy(),
    ):
        raise ValueError(
            "ECGDiff prepared manifest patient identities are not row-aligned"
        )
    if not np.array_equal(
        prepared["split"].astype(str).to_numpy(),
        selected["split"].astype(str).to_numpy(),
    ):
        raise ValueError("ECGDiff prepared manifest split roles are not row-aligned")

    conditioning = np.load(paths["dense_conditioning"], mmap_mode="r")
    if conditioning.shape != (len(selected), 1539) or conditioning.dtype != np.float32:
        raise ValueError(
            f"released conditioning cache must have shape {(len(selected), 1539)} float32; "
            f"found {conditioning.shape} {conditioning.dtype}"
        )
    sex = np.asarray(conditioning[:, 1536], dtype=np.float32)
    age = np.asarray(conditioning[:, 1537], dtype=np.float32)
    if not np.isin(sex, [0.0, 1.0]).all() or not np.isfinite(age).all():
        raise ValueError("released sex/age conditions are invalid")

    selected = selected.reset_index(drop=False).rename(
        columns={"index": "author_selection_index"}
    )
    selected.insert(0, "row_idx", np.arange(len(selected), dtype=np.int64))
    selected["subject_id"] = selected["subject_id"].astype(np.int64)
    selected["study_id"] = selected["study_id"].astype(np.int64)
    selected["packed_idx"] = selected["packed_idx"].astype(np.int64)
    selected["source_manifest_row"] = selected["manifest_row"].astype(np.int64)
    selected["sex_male"] = sex
    selected["age_years"] = age
    selected["rr_interval_ms"] = pd.to_numeric(selected["rr_interval"], errors="coerce")
    selected["conditioning_row"] = selected["row_idx"]
    output_columns = [
        "row_idx",
        "author_selection_index",
        "source_manifest_row",
        "packed_idx",
        "subject_id",
        "study_id",
        "waveform_path",
        "split",
        "report",
        "rr_interval_ms",
        "sex_male",
        "age_years",
        "conditioning_row",
    ]
    table = pa.Table.from_pandas(selected[output_columns], preserve_index=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    pq.write_table(table, temporary, compression="zstd", use_dictionary=["split"])
    temporary.replace(output_path)

    role_counts: dict[str, dict[str, int]] = {}
    patient_sets: dict[str, set[int]] = {}
    for role in ("train", "val", "test"):
        part = selected.loc[selected["split"].eq(role)]
        patients = set(part["subject_id"].astype(int))
        patient_sets[role] = patients
        role_counts[role] = {"records": len(part), "patients": len(patients)}
    overlap = {
        "train_val": len(patient_sets["train"] & patient_sets["val"]),
        "train_test": len(patient_sets["train"] & patient_sets["test"]),
        "val_test": len(patient_sets["val"] & patient_sets["test"]),
    }
    if any(overlap.values()):
        raise ValueError(f"patient contamination detected: {overlap}")
    expected_counts = data.get("expected_counts")
    if expected_counts and role_counts != expected_counts:
        raise ValueError(
            f"role counts differ from the frozen ECGDiff suite: {role_counts}"
        )

    identity_payload = selected[["row_idx", "study_id", "subject_id", "split"]].to_dict(
        "records"
    )
    audit: dict[str, Any] = {
        "schema": "diffusets_patient_disjoint_manifest_v1",
        "records": len(selected),
        "patients": int(selected["subject_id"].nunique()),
        "source_merged_records": source_merged_records,
        "author_exclusion_count": len(exclusions),
        "quality_excluded_record_ids": sorted(quality_excluded_ids),
        "split_policy": {
            "method": "md5_subject_id_seed_first_12_hex",
            "seed": split_seed,
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "test_fraction": 1.0 - train_fraction - validation_fraction,
        },
        "roles": role_counts,
        "patient_overlap": overlap,
        "split_mismatches": split_mismatches,
        "identity_sha256": sha256_json(identity_payload),
        "source_sha256": source_hashes,
        "manifest": str(output_path),
        "manifest_sha256": sha256_file(output_path),
        "conditioning_policy": {
            "text": "released frozen ada-002 embedding; no fitted local parameters",
            "sex_age": "released DiffuSETS conditions",
            "heart_rate": "300--1500 ms RR gate, then raw-waveform WFDB XQRS fallback",
        },
        "test_or_validation_used_for_training_selection": False,
    }
    atomic_json_dump(audit, audit_path)
    return audit


def load_manifest(path: Path) -> pd.DataFrame:
    frame = pq.read_table(path).to_pandas()
    required = {
        "row_idx",
        "subject_id",
        "study_id",
        "split",
        "waveform_path",
        "rr_interval_ms",
        "conditioning_row",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"patient-disjoint manifest is missing columns: {missing}")
    if not np.array_equal(
        frame["row_idx"].to_numpy(dtype=np.int64), np.arange(len(frame))
    ):
        raise ValueError(
            "patient-disjoint manifest row_idx must be contiguous and row-aligned"
        )
    if frame["study_id"].duplicated().any():
        raise ValueError("patient-disjoint manifest contains duplicate records")
    by_patient = frame.groupby("subject_id", sort=False)["split"].nunique()
    if bool(by_patient.gt(1).any()):
        raise ValueError(
            "patient-disjoint manifest assigns a patient to multiple roles"
        )
    return frame


def audit_training_exposure(
    manifest: pd.DataFrame, positions: np.ndarray, *, purpose: str
) -> None:
    roles = set(manifest.iloc[positions]["split"].astype(str))
    if purpose in TRAINING_PURPOSES and roles != {"train"}:
        raise ValueError(
            f"{purpose} may consume only train records; found roles {sorted(roles)}"
        )


class DiffuSETSDataset(Dataset[dict[str, torch.Tensor]]):
    """Memory-mapped clean waveform/latent/condition dataset."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        split: Role,
        purpose: str,
        waveform_path: Path | None = None,
        latent_path: Path | None = None,
        conditioning_path: Path | None = None,
        heart_rate_path: Path | None = None,
        status_path: Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.frame = load_manifest(self.manifest_path)
        self.positions = (
            np.arange(len(self.frame), dtype=np.int64)
            if split == "all"
            else np.flatnonzero(
                self.frame["split"].astype(str).eq(split).to_numpy()
            ).astype(np.int64)
        )
        if status_path is not None:
            status = np.load(status_path, mmap_mode="r")
            if status.shape != (len(self.frame),):
                raise ValueError("waveform status array is not manifest-aligned")
            invalid = self.positions[np.asarray(status[self.positions]) != 1]
            if len(invalid):
                raise ValueError(
                    f"{purpose} requires a complete clean cache; "
                    f"{len(invalid)} selected rows do not have status 1"
                )
        if len(self.positions) == 0:
            raise ValueError(f"dataset split {split!r} is empty")
        audit_training_exposure(self.frame, self.positions, purpose=purpose)
        self.split = split
        self.purpose = purpose
        self.waveform_path = Path(waveform_path) if waveform_path else None
        self.latent_path = Path(latent_path) if latent_path else None
        self.conditioning_path = Path(conditioning_path) if conditioning_path else None
        self.heart_rate_path = Path(heart_rate_path) if heart_rate_path else None
        self._waveforms: np.ndarray[Any, Any] | None = None
        self._latents: np.ndarray[Any, Any] | None = None
        self._conditioning: np.ndarray[Any, Any] | None = None
        self._heart_rate: np.ndarray[Any, Any] | None = None

    def __len__(self) -> int:
        return len(self.positions)

    def _arrays(self) -> tuple[np.ndarray[Any, Any] | None, ...]:
        if self.waveform_path is not None and self._waveforms is None:
            self._waveforms = np.load(self.waveform_path, mmap_mode="r")
            if self._waveforms.shape != (len(self.frame), 1024, 12):
                raise ValueError(
                    f"invalid waveform cache shape {self._waveforms.shape}"
                )
        if self.latent_path is not None and self._latents is None:
            self._latents = np.load(self.latent_path, mmap_mode="r")
            if self._latents.shape != (len(self.frame), 4, 128):
                raise ValueError(f"invalid latent cache shape {self._latents.shape}")
        if self.conditioning_path is not None and self._conditioning is None:
            self._conditioning = np.load(self.conditioning_path, mmap_mode="r")
            if self._conditioning.shape != (len(self.frame), 1539):
                raise ValueError(
                    f"invalid conditioning cache shape {self._conditioning.shape}"
                )
        if self.heart_rate_path is not None and self._heart_rate is None:
            self._heart_rate = np.load(self.heart_rate_path, mmap_mode="r")
            if self._heart_rate.shape != (len(self.frame),):
                raise ValueError(
                    f"invalid heart-rate cache shape {self._heart_rate.shape}"
                )
        return self._waveforms, self._latents, self._conditioning, self._heart_rate

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row_idx = int(self.positions[index])
        waveforms, latents, conditioning, heart_rate = self._arrays()
        item: dict[str, torch.Tensor] = {
            "row_idx": torch.tensor(row_idx, dtype=torch.int64)
        }
        if waveforms is not None:
            item["waveform"] = torch.from_numpy(
                np.array(waveforms[row_idx], dtype=np.float32, copy=True)
            )
        if latents is not None:
            item["latent"] = torch.from_numpy(
                np.array(latents[row_idx], dtype=np.float32, copy=True)
            )
        if conditioning is not None:
            values = np.array(conditioning[row_idx], dtype=np.float32, copy=True)
            item["text_embedding"] = torch.from_numpy(values[:1536])
            item["gender"] = torch.tensor(values[1536], dtype=torch.float32)
            item["age"] = torch.tensor(values[1537], dtype=torch.float32)
        if heart_rate is not None:
            value = float(heart_rate[row_idx])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"row {row_idx} has invalid corrected heart rate {value}"
                )
            item["heart_rate"] = torch.tensor(value, dtype=torch.float32)
        return item


class PaddedDistributedInferenceSampler(Sampler[tuple[int, bool]]):
    """Equal-length rank shards with explicit padding flags for collective inference."""

    def __init__(self, dataset_size: int, rank: int, world_size: int) -> None:
        if dataset_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("invalid distributed inference sampler dimensions")
        self.dataset_size = dataset_size
        self.rank = rank
        self.world_size = world_size
        self.samples_per_rank = math.ceil(dataset_size / world_size)

    def __len__(self) -> int:
        return self.samples_per_rank

    def __iter__(self) -> Iterator[tuple[int, bool]]:
        owned = list(range(self.rank, self.dataset_size, self.world_size))
        for index in owned:
            yield index, True
        padding_index = owned[-1] if owned else 0
        for _ in range(self.samples_per_rank - len(owned)):
            yield padding_index, False


class IndexedDataset(Dataset[dict[str, torch.Tensor]]):
    """Teach a map dataset to preserve whether a collective-inference row is padding."""

    def __init__(self, dataset: Dataset[dict[str, torch.Tensor]]) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int | tuple[int, bool]) -> dict[str, torch.Tensor]:
        if isinstance(index, tuple):
            item = self.dataset[index[0]]
            item["valid"] = torch.tensor(index[1], dtype=torch.bool)
            return item
        item = self.dataset[index]
        item["valid"] = torch.tensor(True, dtype=torch.bool)
        return item


def role_counts(
    frame: pd.DataFrame, status: np.ndarray[Any, Any] | None = None
) -> dict[str, Any]:
    usable = (
        np.ones(len(frame), dtype=bool) if status is None else np.asarray(status) == 1
    )
    result: dict[str, Any] = {}
    for role in ("train", "val", "test"):
        mask = frame["split"].astype(str).eq(role).to_numpy() & usable
        result[role] = {
            "records": int(mask.sum()),
            "patients": int(frame.loc[mask, "subject_id"].nunique()),
        }
    return result
