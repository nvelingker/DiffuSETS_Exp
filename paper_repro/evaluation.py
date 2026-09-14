from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from paper_repro.checkpoint import load_portable_payload
from paper_repro.common import atomic_json_dump, sha256_file
from paper_repro.config import ReproConfig


SCHEMA = "sediff_reconstruction_v2"
ACTIVE_SUITE_ID = "diffusets-clean-patient-disjoint-seed2026-v1"
ACTIVE_CONFIG_SHA256 = "b4f2529a28aa947b050a5d6f8e5c8d4271a90ff13169445b13e8b734adcf8827"
ACTIVE_MANIFEST_SHA256 = "a51858be5dd513dd78cf34dfd2cf58b1f166301404ef7a0da4740b39bfa103bc"
ACTIVE_LATENT_SHA256 = "ccd1bb70ec720d296f9f7ba52480ac72f911c9fabeb117906d22e2eb769fa99d"
ACTIVE_HEART_RATE_SHA256 = "7dbebb9c3d00314fa476ae00ae5237fb8fb2840d621120c5ad341dc9f366a4a7"
ACTIVE_VAE_SHA256 = "5ade85ed4ff7bfac0b6f5196785c2d452d4279e458f59e54fc34242cf5b3ea0a"
ACTIVE_CLIP_SHA256 = "45774181d91b61b72d78bfa1add3a59572680a1fdc5737d31a293f521fcbea64"
ACTIVE_UNET_SHA256 = "240dc61fa40d6eaa7db21e29757d368ccf9636d6341d0b36525b8595876747aa"
ACTIVE_PTBXL_PACKAGE_SHA256 = (
    "00e48314b82d10bcf56993b895b33c5b5bf2f8dd9d03017c8129bef9df4e7be4"
)

DIFFUSETS_LEADS = (
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
)
CANONICAL_LEADS = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)
CANONICAL_FROM_DIFFUSETS = tuple(DIFFUSETS_LEADS.index(lead) for lead in CANONICAL_LEADS)
DIFFUSETS_FROM_CANONICAL = tuple(CANONICAL_LEADS.index(lead) for lead in DIFFUSETS_LEADS)


@dataclass(frozen=True, slots=True)
class EvaluationConditions:
    frame: pd.DataFrame
    text: np.ndarray
    metadata: np.ndarray
    source_rows: np.ndarray
    summary: dict[str, Any]
    summary_path: Path
    summary_sha256: str
    text_path: Path
    text_sha256: str
    provenance: dict[str, Any]


def require_sha256(path: Path, expected: str, label: str) -> str:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, found {actual}")
    return actual


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON value is not an object: {path}")
    return value


def _verified_summary_file(
    summary: dict[str, Any], summary_path: Path, key: str
) -> Path:
    value = summary.get(key)
    expected = summary.get(f"{key}_sha256")
    if not isinstance(value, str) or not isinstance(expected, str):
        raise ValueError(f"condition summary lacks a hash-bound {key}")
    path = Path(value).expanduser().resolve()
    require_sha256(path, expected, f"condition {key}")
    return path


def load_condition_envelope(directory: Path) -> tuple[dict[str, Any], Path, str, pd.DataFrame]:
    summary_path = (directory.expanduser().resolve() / "summary.json")
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = _read_json(summary_path)
    if (
        summary.get("schema") != SCHEMA
        or summary.get("stage") != "evaluation_condition_package"
        or summary.get("status") != "frozen"
    ):
        raise ValueError(f"not a frozen evaluation condition package: {summary_path}")
    manifest_path = _verified_summary_file(summary, summary_path, "manifest")
    _verified_summary_file(summary, summary_path, "reference_ecg")
    frame = pd.read_csv(manifest_path, low_memory=False)
    records = summary.get("records")
    if isinstance(records, bool) or not isinstance(records, int) or records < 1:
        raise ValueError("condition summary has an invalid record count")
    if len(frame) != records:
        raise ValueError("condition manifest length differs from its summary")
    if "condition_idx" not in frame or not np.array_equal(
        pd.to_numeric(frame["condition_idx"], errors="raise").to_numpy(np.int64),
        np.arange(records, dtype=np.int64),
    ):
        raise ValueError("condition_idx must be contiguous, zero based, and row aligned")
    if "study_id" not in frame or frame["study_id"].astype(str).duplicated().any():
        raise ValueError("condition manifest study IDs must be unique")
    return summary, summary_path, sha256_file(summary_path), frame


def _validate_text(values: np.ndarray, count: int) -> np.ndarray:
    text = np.asarray(values, dtype=np.float32)
    if text.shape != (count, 1536) or not np.isfinite(text).all():
        raise ValueError(f"text conditions must be finite {(count, 1536)} float32")
    return np.ascontiguousarray(text)


def load_mimic_conditions(config: ReproConfig, directory: Path) -> EvaluationConditions:
    summary, summary_path, summary_sha, frame = load_condition_envelope(directory)
    clean_manifest_path = config.paths["manifest"]
    require_sha256(clean_manifest_path, ACTIVE_MANIFEST_SHA256, "clean manifest")
    clean = pq.read_table(clean_manifest_path).to_pandas()
    if clean["study_id"].duplicated().any():
        raise ValueError("clean DiffuSETS manifest has duplicate study IDs")
    positions = pd.Series(
        clean["row_idx"].to_numpy(np.int64), index=clean["study_id"].astype(str)
    ).reindex(frame["study_id"].astype(str))
    if positions.isna().any() or positions.nunique() != len(frame):
        raise ValueError("MIMIC comparison rows do not map one-to-one into the clean manifest")
    source_rows = positions.to_numpy(np.int64)
    selected = clean.iloc[source_rows].reset_index(drop=True)
    if set(selected["split"].astype(str)) != {"test"}:
        raise ValueError("MIMIC comparison contains a non-test clean DiffuSETS row")
    panel_subject_column = "subject_id" if "subject_id" in frame else "patient_id"
    if panel_subject_column not in frame or not np.array_equal(
        selected["subject_id"].astype(str).to_numpy(),
        frame[panel_subject_column].astype(str).to_numpy(),
    ):
        raise ValueError("MIMIC panel patient identities differ from the clean manifest")

    dense_path = config.paths["dense_conditioning"]
    dense_expected = str(config.section("data")["source_sha256"]["dense_conditioning"])
    require_sha256(dense_path, dense_expected, "clean dense conditioning")
    dense = np.load(dense_path, mmap_mode="r", allow_pickle=False)
    if dense.shape != (len(clean), 1539) or dense.dtype != np.float32:
        raise ValueError("clean dense conditioning has the wrong shape or dtype")
    text = _validate_text(dense[source_rows, :1536], len(frame))
    summary_text_path = _verified_summary_file(summary, summary_path, "text_embeddings")
    summary_text = np.load(summary_text_path, mmap_mode="r", allow_pickle=False)
    if not np.array_equal(text, summary_text):
        raise ValueError("frozen MIMIC panel text differs from clean DiffuSETS conditioning")

    heart_rate_path = config.paths["heart_rate"]
    require_sha256(
        heart_rate_path, ACTIVE_HEART_RATE_SHA256, "clean repaired heart rate"
    )
    heart_rate = np.load(heart_rate_path, mmap_mode="r", allow_pickle=False)
    metadata = np.stack(
        (
            np.asarray(dense[source_rows, 1536], dtype=np.float32),
            np.asarray(dense[source_rows, 1537], dtype=np.float32),
            np.asarray(heart_rate[source_rows], dtype=np.float32),
        ),
        axis=1,
    )
    if metadata.shape != (len(frame), 3) or not np.isfinite(metadata).all():
        raise ValueError("clean MIMIC sex/age/heart-rate conditions are malformed")
    if not np.isin(metadata[:, 0], (0.0, 1.0)).all() or np.any(metadata[:, 1:] <= 0):
        raise ValueError("clean MIMIC scalar conditions are outside their valid domains")
    shared_hr = pd.to_numeric(frame.get("hr_bpm"), errors="coerce").to_numpy(np.float64)
    finite_shared = np.isfinite(shared_hr)
    return EvaluationConditions(
        frame=frame,
        text=text,
        metadata=np.ascontiguousarray(metadata),
        source_rows=source_rows,
        summary=summary,
        summary_path=summary_path,
        summary_sha256=summary_sha,
        text_path=summary_text_path,
        text_sha256=sha256_file(summary_text_path),
        provenance={
            "conditioning_source": "clean patient-disjoint manifest and repaired-RR cache",
            "clean_manifest": str(clean_manifest_path),
            "clean_manifest_sha256": ACTIVE_MANIFEST_SHA256,
            "clean_roles": {"test": len(frame)},
            "patients": int(selected["subject_id"].nunique()),
            "text_matches_frozen_panel_exactly": True,
            "shared_hr_finite_records": int(finite_shared.sum()),
            "clean_hr_vs_shared_hr_mae_bpm": (
                float(np.mean(np.abs(metadata[finite_shared, 2] - shared_hr[finite_shared])))
                if finite_shared.any()
                else None
            ),
        },
    )


def load_ptbxl_conditions(
    directory: Path,
    *,
    text_path: Path,
    text_sha256: str,
    package_path: Path,
) -> EvaluationConditions:
    summary, summary_path, summary_sha, frame = load_condition_envelope(directory)
    require_sha256(text_path, text_sha256, "fixed PTB-XL DiffuSETS text embeddings")
    text = _validate_text(
        np.load(text_path, mmap_mode="r", allow_pickle=False), len(frame)
    )
    require_sha256(package_path, ACTIVE_PTBXL_PACKAGE_SHA256, "PTB-XL condition package")
    package = torch.load(
        package_path, map_location="cpu", weights_only=False, mmap=True
    )
    if not isinstance(package, dict) or "packed_idx" not in frame:
        raise ValueError("PTB-XL condition package or manifest is malformed")
    source_rows = pd.to_numeric(frame["packed_idx"], errors="raise").to_numpy(np.int64)
    metadata_rows: list[tuple[float, float, float]] = []
    for position in source_rows:
        try:
            label = package[int(position)]["label"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"PTB-XL package lacks condition row {position}") from error
        if not isinstance(label, dict):
            raise ValueError(f"PTB-XL package label {position} is malformed")
        gender = str(label.get("gender", "")).strip().upper()
        if gender not in {"F", "M"}:
            raise ValueError(f"PTB-XL package gender {position} is invalid")
        age = float(label.get("age"))
        heart_rate = float(label.get("hr"))
        if not math.isfinite(age) or age <= 0 or not math.isfinite(heart_rate) or heart_rate <= 0:
            raise ValueError(f"PTB-XL package scalar condition {position} is invalid")
        metadata_rows.append((1.0 if gender == "M" else 0.0, age, heart_rate))
    metadata = np.asarray(metadata_rows, dtype=np.float32)
    manifest_age = pd.to_numeric(frame.get("age"), errors="coerce").to_numpy(np.float64)
    manifest_hr = pd.to_numeric(frame.get("hr_bpm"), errors="coerce").to_numpy(np.float64)
    finite_age = np.isfinite(manifest_age)
    finite_hr = np.isfinite(manifest_hr)
    if not np.allclose(metadata[finite_age, 1], manifest_age[finite_age], rtol=0, atol=0):
        raise ValueError("finite PTB-XL ages differ from the author condition package")
    if "sex" in frame:
        expected_sex = frame["sex"].astype(str).str.upper().map({"F": 0.0, "M": 1.0})
        if expected_sex.isna().any() or not np.array_equal(
            metadata[:, 0], expected_sex.to_numpy(np.float32)
        ):
            raise ValueError("PTB-XL sex differs from the author condition package")
    return EvaluationConditions(
        frame=frame,
        text=text,
        metadata=np.ascontiguousarray(metadata),
        source_rows=source_rows,
        summary=summary,
        summary_path=summary_path,
        summary_sha256=summary_sha,
        text_path=text_path.expanduser().resolve(),
        text_sha256=text_sha256,
        provenance={
            "conditioning_source": "author PTB-XL data package in fixed local cohort order",
            "ptbxl_package": str(package_path.expanduser().resolve()),
            "ptbxl_package_sha256": ACTIVE_PTBXL_PACKAGE_SHA256,
            "text_embedding_source": "recovered saved DiffuSETS PTB-XL conditions",
            "text_zero_embedding_count": int(np.all(text == 0, axis=1).sum()),
            "author_age_sentinel_records": int((metadata[:, 1] == 300.0).sum()),
            "shared_age_missing_records": int((~finite_age).sum()),
            "shared_hr_missing_records": int((~finite_hr).sum()),
            "author_hr_vs_shared_hr_mae_bpm_on_finite_shared": (
                float(np.mean(np.abs(metadata[finite_hr, 2] - manifest_hr[finite_hr])))
                if finite_hr.any()
                else None
            ),
        },
    )


def validate_active_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    expected_stage: str,
    config: ReproConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require_sha256(path, expected_sha256, f"active {expected_stage} checkpoint")
    state, provenance = load_portable_payload(path)
    if provenance.get("schema") != "diffusets_clean_checkpoint_v1":
        raise ValueError(f"active {expected_stage} checkpoint schema changed")
    if provenance.get("stage") != expected_stage:
        raise ValueError(f"active checkpoint is not a {expected_stage} checkpoint")
    if provenance.get("config_sha256") != ACTIVE_CONFIG_SHA256:
        raise ValueError(f"active {expected_stage} checkpoint config binding changed")
    if provenance.get("manifest_sha256") != ACTIVE_MANIFEST_SHA256:
        raise ValueError(f"active {expected_stage} checkpoint manifest binding changed")
    if config.config_sha256 != ACTIVE_CONFIG_SHA256:
        raise ValueError("evaluation configuration differs from the registered clean suite")
    return state, provenance


def active_suite_provenance(config: ReproConfig) -> dict[str, Any]:
    paths = config.paths
    _, vae = validate_active_checkpoint(
        paths["vae_checkpoint"],
        expected_sha256=ACTIVE_VAE_SHA256,
        expected_stage="vae",
        config=config,
    )
    _, clip = validate_active_checkpoint(
        paths["clip_checkpoint"],
        expected_sha256=ACTIVE_CLIP_SHA256,
        expected_stage="clip",
        config=config,
    )
    _, unet = validate_active_checkpoint(
        paths["diffusion_checkpoint"],
        expected_sha256=ACTIVE_UNET_SHA256,
        expected_stage="diffusion",
        config=config,
    )
    if clip.get("vae_checkpoint_sha256") != ACTIVE_VAE_SHA256:
        raise ValueError("clean CLIP does not bind the active VAE")
    if unet.get("latent_sha256") != ACTIVE_LATENT_SHA256:
        raise ValueError("clean U-Net does not bind the active latent cache")
    return {
        "suite_id": ACTIVE_SUITE_ID,
        "config": str(config.path),
        "config_sha256": ACTIVE_CONFIG_SHA256,
        "manifest": str(paths["manifest"]),
        "manifest_sha256": ACTIVE_MANIFEST_SHA256,
        "vae_checkpoint": str(paths["vae_checkpoint"]),
        "vae_checkpoint_sha256": ACTIVE_VAE_SHA256,
        "clip_checkpoint": str(paths["clip_checkpoint"]),
        "clip_checkpoint_sha256": ACTIVE_CLIP_SHA256,
        "diffusion_checkpoint": str(paths["diffusion_checkpoint"]),
        "diffusion_checkpoint_sha256": ACTIVE_UNET_SHA256,
        "latent_sha256": ACTIVE_LATENT_SHA256,
        "training_git": unet.get("git"),
        "training_epoch": unet.get("epoch"),
        "training_global_step": unet.get("global_step"),
    }


def repository_state(root: Path) -> dict[str, Any]:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return {"head": "unavailable", "dirty": True}
    return {"head": head, "dirty": bool(status.strip())}


def save_array(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)
    return sha256_file(path)


def canonicalize_decoded(waveforms: torch.Tensor) -> torch.Tensor:
    if waveforms.ndim != 3 or tuple(waveforms.shape[1:]) != (1024, 12):
        raise ValueError("DiffuSETS decoder output must be [batch,1024,12]")
    return waveforms.transpose(1, 2)[:, CANONICAL_FROM_DIFFUSETS].contiguous()


def diffusets_order_from_canonical(waveforms: torch.Tensor) -> torch.Tensor:
    if waveforms.ndim != 3 or tuple(waveforms.shape[1:]) != (12, 1024):
        raise ValueError("canonical waveform input must be [batch,12,1024]")
    return waveforms[:, DIFFUSETS_FROM_CANONICAL].transpose(1, 2).contiguous()


def condition_identity_sha256(conditions: EvaluationConditions) -> str:
    payload = {
        "study_ids": conditions.frame["study_id"].astype(str).tolist(),
        "source_rows": conditions.source_rows.tolist(),
        "text_sha256": hashlib.sha256(conditions.text.tobytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(conditions.metadata.tobytes()).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


__all__ = [
    "ACTIVE_CLIP_SHA256",
    "ACTIVE_CONFIG_SHA256",
    "ACTIVE_HEART_RATE_SHA256",
    "ACTIVE_MANIFEST_SHA256",
    "ACTIVE_PTBXL_PACKAGE_SHA256",
    "ACTIVE_SUITE_ID",
    "ACTIVE_UNET_SHA256",
    "ACTIVE_VAE_SHA256",
    "CANONICAL_FROM_DIFFUSETS",
    "CANONICAL_LEADS",
    "DIFFUSETS_FROM_CANONICAL",
    "DIFFUSETS_LEADS",
    "EvaluationConditions",
    "SCHEMA",
    "active_suite_provenance",
    "atomic_json_dump",
    "canonicalize_decoded",
    "condition_identity_sha256",
    "diffusets_order_from_canonical",
    "load_condition_envelope",
    "load_mimic_conditions",
    "load_ptbxl_conditions",
    "repository_state",
    "require_sha256",
    "save_array",
    "sha256_file",
    "validate_active_checkpoint",
]
