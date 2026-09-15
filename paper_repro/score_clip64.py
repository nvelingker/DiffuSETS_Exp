from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.linalg import sqrtm

from clip.clip_model import CLIP
from paper_repro.config import load_config
from paper_repro.evaluation import (
    ACTIVE_BASE_CONFIG_SHA256,
    ACTIVE_CLIP_SHA256,
    ACTIVE_SUITE_ID,
    CANONICAL_LEADS,
    DIFFUSETS_LEADS,
    SCHEMA,
    atomic_json_dump,
    diffusets_order_from_canonical,
    load_condition_envelope,
    require_sha256,
    sha256_file,
    validate_active_checkpoint,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config/patient_disjoint_fsdp2.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a canonical waveform panel with clean DiffuSETS-CLIP64 seed2026."
    )
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-embeddings", type=Path, default=None)
    parser.add_argument("--text-embeddings-sha256", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON is not an object: {path}")
    return value


@torch.inference_mode()
def encode(
    model: torch.nn.Module,
    waveforms: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for start in range(0, len(waveforms), batch_size):
        canonical = torch.from_numpy(
            np.array(waveforms[start : start + batch_size], dtype=np.float32, copy=True)
        ).to(device)
        training_order = diffusets_order_from_canonical(canonical)
        rows.append(model.ecg_projector(model.encode_signal(training_order)).float().cpu())
    result = torch.cat(rows)
    if result.shape != (len(waveforms), 64) or not bool(torch.isfinite(result).all()):
        raise ValueError("clean DiffuSETS-CLIP64 ECG features are malformed")
    return result


def fid_score(first: torch.Tensor, second: torch.Tensor) -> float:
    one = first.numpy().astype(np.float64, copy=False)
    two = second.numpy().astype(np.float64, copy=False)
    mean_one, covariance_one = one.mean(axis=0), np.cov(one, rowvar=False)
    mean_two, covariance_two = two.mean(axis=0), np.cov(two, rowvar=False)
    covariance_mean = sqrtm(covariance_one.dot(covariance_two))
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    value = float(
        np.sum((mean_one - mean_two) ** 2)
        + np.trace(covariance_one + covariance_two - 2.0 * covariance_mean)
    )
    if not math.isfinite(value):
        raise FloatingPointError("clean DiffuSETS-CLIP64 FID is nonfinite")
    return value


def radii(features: torch.Tensor, k: int = 3) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for start in range(0, len(features), 512):
        distance = torch.cdist(features[start : start + 512], features)
        rows.append(torch.topk(distance, k=k + 1, dim=1, largest=False).values[:, -1])
    return torch.cat(rows)


def manifold_fraction(
    test: torch.Tensor, manifold: torch.Tensor, manifold_radii: torch.Tensor
) -> float:
    inside = 0
    for start in range(0, len(test), 512):
        distance = torch.cdist(test[start : start + 512], manifold)
        inside += int((distance <= manifold_radii).any(dim=1).sum())
    return inside / len(test)


def _text_artifact(
    args: argparse.Namespace, inference: dict[str, Any], conditions: dict[str, Any]
) -> tuple[Path, str]:
    if args.text_embeddings is not None:
        if not isinstance(args.text_embeddings_sha256, str):
            raise ValueError("an explicit text embedding file requires its SHA-256")
        return args.text_embeddings.expanduser().resolve(), args.text_embeddings_sha256
    path_value = inference.get("text_embeddings", conditions.get("text_embeddings"))
    hash_value = inference.get(
        "text_embeddings_sha256", conditions.get("text_embeddings_sha256")
    )
    if not isinstance(path_value, str) or not isinstance(hash_value, str):
        raise ValueError("inference/condition metadata does not bind text embeddings")
    return Path(path_value).expanduser().resolve(), hash_value


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    inference_dir = args.inference_dir.expanduser().resolve()
    inference_path = inference_dir / "summary.json"
    if not inference_path.is_file():
        raise FileNotFoundError(inference_path)
    inference = _read_json(inference_path)
    if (
        inference.get("schema") != SCHEMA
        or inference.get("stage") != "ddpm_cfg_inference"
        or inference.get("status") != "complete"
    ):
        raise ValueError("input is not a completed shared inference artifact")
    conditions, condition_path, condition_sha, frame = load_condition_envelope(
        Path(inference["condition_dir"])
    )
    if condition_sha != inference.get("condition_summary_sha256"):
        raise ValueError("inference condition-summary binding changed")
    generated_path = Path(inference["waveforms"])
    reference_path = Path(conditions["reference_ecg"])
    require_sha256(generated_path, str(inference["waveforms_sha256"]), "generated waveforms")
    require_sha256(
        reference_path, str(conditions["reference_ecg_sha256"]), "reference waveforms"
    )
    generated = np.load(generated_path, mmap_mode="r", allow_pickle=False)
    reference = np.load(reference_path, mmap_mode="r", allow_pickle=False)
    draws, records = generated.shape[:2]
    if generated.shape != (draws, records, 12, 1024) or reference.shape != (
        records,
        12,
        1024,
    ):
        raise ValueError("generated/reference panel geometry changed")
    if len(frame) != records:
        raise ValueError("condition rows differ from generated panel")
    if not np.isfinite(generated).all() or not np.isfinite(reference).all():
        raise ValueError("learned-score inputs contain nonfinite waveforms")

    text_path, text_sha = _text_artifact(args, inference, conditions)
    require_sha256(text_path, text_sha, "DiffuSETS text embeddings")
    text = np.load(text_path, mmap_mode="r", allow_pickle=False)
    if text.shape != (records, 1536) or not np.isfinite(text).all():
        raise ValueError("text embeddings are not finite [record,1536]")

    config = load_config(args.config)
    if config.config_sha256 != ACTIVE_BASE_CONFIG_SHA256:
        raise ValueError("CLIP scoring config is not the active clean-suite base config")
    state, provenance = validate_active_checkpoint(
        config.paths["clip_checkpoint"],
        expected_sha256=ACTIVE_CLIP_SHA256,
        expected_stage="clip",
        expected_config_sha256=ACTIVE_BASE_CONFIG_SHA256,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = CLIP(embed_dim=64)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval().to(device)
    generated_features = encode(
        model, generated.reshape(-1, 12, 1024), args.batch_size, device
    )
    reference_features = encode(model, reference, args.batch_size, device)
    fid = fid_score(reference_features, generated_features)
    midpoint = records // 2
    real_split_fid = fid_score(
        reference_features[:midpoint], reference_features[midpoint:]
    )
    generated_device = generated_features.to(device)
    reference_device = reference_features.to(device)
    generated_radii = radii(generated_device, 3)
    reference_radii = radii(reference_device, 3)
    precision = manifold_fraction(generated_device, reference_device, reference_radii)
    recall = manifold_fraction(reference_device, generated_device, generated_radii)
    with torch.inference_mode():
        text_tensor = torch.from_numpy(np.array(text, dtype=np.float32, copy=True)).to(device)
        text_features = model.text_projector(text_tensor)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        generated_normalized = generated_device / generated_device.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        reference_normalized = reference_device / reference_device.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        repeated_text = text_features.repeat((draws, 1))
        text_cosine = float((generated_normalized * repeated_text).sum(dim=-1).mean())
        reference_text_cosine = float(
            (reference_normalized * text_features).sum(dim=-1).mean()
        )
    rclip = (
        text_cosine / reference_text_cosine
        if abs(reference_text_cosine) > 1e-12
        else None
    )
    legacy_rfid = fid / real_split_fid
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    report = {
        "schema": SCHEMA,
        "stage": "bespoke_learned_metric_sensitivity",
        "status": "complete",
        "metric_family": "DiffuSETS-clean-CLIP64 seed2026",
        "suite_id": ACTIVE_SUITE_ID,
        "primary_model_agnostic_metric": False,
        "warning": (
            "This learned evaluator is the validation-selected component of the local "
            "patient-disjoint DiffuSETS reproduction. It is a bespoke sensitivity metric, "
            "not independent ground truth."
        ),
        "dataset": inference["dataset"],
        "records": records,
        "samples_per_condition": draws,
        "metrics": {
            "fid": fid,
            "precision_k3": precision,
            "recall_k3": recall,
            "manifold_f1_k3": f1,
            "text_cosine": text_cosine,
            "reference_text_cosine": reference_text_cosine,
            "rclip_ratio_of_means": rclip,
            "real_split_fid_contiguous_halves": real_split_fid,
            "rfid_arxiv_v1_generated_real_over_real_split": legacy_rfid,
            "rfid_iclr2026_real_split_over_sum": 1.0 / (1.0 + legacy_rfid),
        },
        "checkpoint": str(config.paths["clip_checkpoint"]),
        "checkpoint_sha256": ACTIVE_CLIP_SHA256,
        "checkpoint_training_roles": provenance.get("training_roles"),
        "checkpoint_selection_roles": provenance.get("selection_roles"),
        "checkpoint_epoch": provenance.get("epoch"),
        "checkpoint_validation_clip": provenance.get("best_validation_clip"),
        "input_lead_order": list(CANONICAL_LEADS),
        "evaluator_training_lead_order": list(DIFFUSETS_LEADS),
        "lead_adapter": "canonical aVL/aVF converted back to DiffuSETS aVF/aVL before encoding",
        "inference_summary": str(inference_path),
        "inference_summary_sha256": sha256_file(inference_path),
        "condition_summary": str(condition_path),
        "condition_summary_sha256": condition_sha,
        "generated_waveforms_sha256": inference["waveforms_sha256"],
        "reference_waveforms_sha256": conditions["reference_ecg_sha256"],
        "text_embeddings": str(text_path),
        "text_embeddings_sha256": text_sha,
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(report, output)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
