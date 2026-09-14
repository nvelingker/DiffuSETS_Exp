from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from paper_repro.common import atomic_json_dump, sha256_file
from paper_repro.evaluation import (
    ACTIVE_CLIP_SHA256,
    ACTIVE_CONFIG_SHA256,
    ACTIVE_SUITE_ID,
    ACTIVE_UNET_SHA256,
    ACTIVE_VAE_SHA256,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATION_ROOT = (
    REPO_ROOT
    / "paper_repro/evaluation/clean_seed2026_e200_base20260822_v1"
)
MODEL_ORDER = ("ecgdiff", "diffusets", "sediff")
MODEL_LABELS = {
    "ecgdiff": "ECGDiff e16",
    "diffusets": "DiffuSETS clean e200",
    "sediff": "SE-Diff v3 e195",
}
EXPECTED_SEDIFF_CHECKPOINT_SHA256 = (
    "acac5f55ca4c12c2427dc150a99be2d3affdeab471c0c054ac9745e816fe7861"
)
EXPECTED_CLIP_IMPLEMENTATION_SHA256 = (
    "307ceee605ad2f04db03da9511ff701ae3180480ea574e55c6d010184e2c231f"
)
EXPECTED_MODEL_AGNOSTIC_IMPLEMENTATION_SHA256 = (
    "ce71a8d730cf2772b21c3ba11dfc96e411ce24a2e6158c421755c80d96eaeb35"
)
EXPECTED_MORPHOLOGY_IMPLEMENTATION_SHA256 = (
    "a2f220796e1d58e110ababfb21c6d17436cadfd4f15f5312cdfea99cd47c95aa"
)
SCORER_ROOT = REPO_ROOT.parent / "SE-Diff"

EXPECTED = {
    "mimic": {
        "dataset": "mimic_released_condition_intersection_2149",
        "records": 2149,
        "patients": 419,
        "condition_summary_sha256": (
            "ef4e804c33d4a0e4bc56570f8c5e27901e720aff2fcee7a3a649f4ab8694c579"
        ),
        "reference_waveforms_sha256": (
            "4db4b73bcbcaf85bb45e67afab32f9ab5bbfcec695e0d0a48af3dc3edd3e4513"
        ),
        "waveforms_sha256": {
            "diffusets": (
                "362e2a8f307e6034c23b20fb3af06f7bc2daa602af40ad41f7a33ac018439a9f"
            ),
            "ecgdiff": (
                "93a437ea8245ae7dda204106167b7cd5c3135ecce60b6aeb9033f3cf58c124af"
            ),
            "sediff": (
                "1c37209d0b883af1b932f7cfa55ff31cfe9a618f3a626aafaff79e06850ddfea"
            ),
        },
    },
    "ptbxl": {
        "dataset": "ptbxl_diffusets_local1000_external_ood",
        "records": 1000,
        "patients": 991,
        "condition_summary_sha256": (
            "2e0f64fd30f01e2a77ec51a929eed453adbff2b146eeb360ee0900520751b1a5"
        ),
        "reference_waveforms_sha256": (
            "f4575b34471245c95d0e709b0b3e8d18e4aa4455ea6b52f318feb82a26ebb7a4"
        ),
        "waveforms_sha256": {
            "diffusets": (
                "f3c0e76c688f6de29e121a49fb0c9783ae9a80fb38b1926c981cfab8b556a672"
            ),
            "ecgdiff": (
                "c0eba0819aa787b7446efed80d1176ab1c6e987a7076699ba0dad10f46b1ca37"
            ),
            "sediff": (
                "5500575e07ede3d76dfc8e88a04b3052a1a361998c6da987a1f1551fa3cf0da2"
            ),
        },
    },
}


@dataclass(frozen=True, slots=True)
class ScoreBundle:
    model_agnostic: dict[str, Any]
    morphology: dict[str, Any]
    clip64: dict[str, Any]
    inference: dict[str, Any]
    paths: dict[str, str]
    hashes: dict[str, str]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and summarize the locked clean DiffuSETS comparison."
    )
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260914)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON is not an object: {path}")
    return value


def _expect(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, found {actual!r}")


def _require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    _expect(actual, expected, f"{label} SHA-256")
    return actual


def _validate_hashed_artifacts(summary: dict[str, Any], label: str) -> None:
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(f"{label} does not bind its output artifacts")
    for name, item in artifacts.items():
        if not isinstance(item, dict):
            raise ValueError(f"{label} artifact {name!r} is malformed")
        path, expected = item.get("path"), item.get("sha256")
        if not isinstance(path, str) or not isinstance(expected, str):
            raise ValueError(f"{label} artifact {name!r} lacks a path/hash")
        _require_hash(Path(path), expected, f"{label} artifact {name}")


def _sediff_checkpoint_sha(inference: dict[str, Any]) -> Any:
    if isinstance(inference.get("source"), dict):
        value = inference["source"].get("checkpoint_sha256")
        if value is not None:
            return value
    return inference.get("diffusion_checkpoint_sha256")


def _validate_model_identity(model: str, inference: dict[str, Any]) -> None:
    if model == "diffusets":
        _expect(inference.get("suite_id"), ACTIVE_SUITE_ID, "DiffuSETS suite")
        _expect(
            inference.get("config_sha256"), ACTIVE_CONFIG_SHA256, "DiffuSETS config"
        )
        _expect(
            inference.get("vae_checkpoint_sha256"),
            ACTIVE_VAE_SHA256,
            "DiffuSETS VAE",
        )
        _expect(
            inference.get("diffusion_checkpoint_sha256"),
            ACTIVE_UNET_SHA256,
            "DiffuSETS U-Net",
        )
        _expect(inference.get("checkpoint_epoch"), 200, "DiffuSETS epoch")
        _expect(inference.get("checkpoint_step"), 54_400, "DiffuSETS step")
        _expect(inference.get("base_seed"), 20_260_822, "DiffuSETS base seed")
        _expect(inference.get("inference_steps"), 1000, "DiffuSETS inference steps")
        _expect(inference.get("guidance_scale"), 1.0, "DiffuSETS guidance scale")
        _expect(inference.get("clip_predicted_x0"), True, "DiffuSETS x0 clipping")
        _expect(
            inference.get("repository_state", {}).get("dirty"),
            False,
            "DiffuSETS source dirtiness",
        )
    elif model == "ecgdiff":
        source = inference.get("source", {})
        _expect(source.get("checkpoint_id"), "step_00004624", "ECGDiff checkpoint")
    elif model == "sediff":
        _expect(
            _sediff_checkpoint_sha(inference),
            EXPECTED_SEDIFF_CHECKPOINT_SHA256,
            "SE-Diff checkpoint",
        )
    else:
        raise ValueError(f"unknown model {model!r}")


def load_bundle(root: Path, dataset: str, model: str) -> ScoreBundle:
    expected = EXPECTED[dataset]
    score_root = root / "scores" / dataset / model
    paths = {
        "model_agnostic": str((score_root / "model_agnostic/metrics.json").resolve()),
        "morphology": str((score_root / "paper_morphology_v3/summary.json").resolve()),
        "clip64": str((score_root / "diffusets_clean_clip64_seed2026.json").resolve()),
    }
    model_agnostic = _read_json(Path(paths["model_agnostic"]))
    morphology = _read_json(Path(paths["morphology"]))
    clip64 = _read_json(Path(paths["clip64"]))

    _expect(model_agnostic.get("schema"), "sediff_reconstruction_v2", "score schema")
    _expect(
        model_agnostic.get("stage"),
        "model_agnostic_evaluation",
        "model-agnostic stage",
    )
    _expect(model_agnostic.get("dataset"), expected["dataset"], "dataset identity")
    _expect(model_agnostic.get("records"), expected["records"], "record count")
    _expect(model_agnostic.get("patients"), expected["patients"], "patient count")
    _expect(model_agnostic.get("samples_per_condition"), 1, "draw count")
    _expect(
        model_agnostic.get("condition_summary_sha256"),
        expected["condition_summary_sha256"],
        "condition summary",
    )
    _expect(
        model_agnostic.get("reference_waveforms_sha256"),
        expected["reference_waveforms_sha256"],
        "reference waveform",
    )
    waveform_sha = expected["waveforms_sha256"][model]
    _expect(
        model_agnostic.get("generated_waveforms_sha256"),
        waveform_sha,
        "generated waveform",
    )
    _expect(model_agnostic.get("bootstrap", {}).get("replicates"), 1000, "score bootstrap")
    _expect(model_agnostic.get("bootstrap", {}).get("seed"), 20260903, "score seed")

    inference_path = Path(str(model_agnostic["inference_summary"]))
    inference_hash = _require_hash(
        inference_path,
        str(model_agnostic["inference_summary_sha256"]),
        f"{dataset}/{model} inference summary",
    )
    inference = _read_json(inference_path)
    _expect(inference.get("schema"), "sediff_reconstruction_v2", "inference schema")
    _expect(inference.get("stage"), "ddpm_cfg_inference", "inference stage")
    _expect(inference.get("status"), "complete", "inference status")
    _expect(inference.get("dataset"), expected["dataset"], "inference dataset")
    _expect(inference.get("records"), expected["records"], "inference records")
    _expect(inference.get("samples_per_condition"), 1, "inference draws")
    _expect(
        inference.get("condition_summary_sha256"),
        expected["condition_summary_sha256"],
        "inference condition summary",
    )
    _expect(inference.get("waveforms_sha256"), waveform_sha, "inference waveform")
    _require_hash(Path(str(inference["waveforms"])), waveform_sha, "waveform array")
    _validate_model_identity(model, inference)

    per_record_path = Path(str(model_agnostic["per_record_metrics"]))
    _require_hash(
        per_record_path,
        str(model_agnostic["per_record_metrics_sha256"]),
        f"{dataset}/{model} per-record scores",
    )

    _expect(morphology.get("schema"), "sediff_reconstruction_v3", "morphology schema")
    _expect(morphology.get("stage"), "paper_morphology_cache", "morphology stage")
    _expect(morphology.get("status"), "complete", "morphology status")
    _expect(morphology.get("records"), expected["records"], "morphology records")
    _expect(morphology.get("samples_per_condition"), 1, "morphology draws")
    _expect(
        morphology.get("condition_summary_sha256"),
        expected["condition_summary_sha256"],
        "morphology condition summary",
    )
    _expect(
        morphology.get("reference_waveforms_sha256"),
        expected["reference_waveforms_sha256"],
        "morphology reference",
    )
    _expect(morphology.get("generated_waveforms_sha256"), waveform_sha, "morphology waveform")
    _expect(
        morphology.get("inference_summary_sha256"),
        inference_hash,
        "morphology inference binding",
    )
    _validate_hashed_artifacts(morphology, f"{dataset}/{model} morphology")

    _expect(clip64.get("schema"), "sediff_reconstruction_v2", "CLIP schema")
    _expect(
        clip64.get("stage"), "bespoke_learned_metric_sensitivity", "CLIP stage"
    )
    _expect(clip64.get("status"), "complete", "CLIP status")
    _expect(clip64.get("suite_id"), ACTIVE_SUITE_ID, "CLIP suite")
    _expect(clip64.get("checkpoint_sha256"), ACTIVE_CLIP_SHA256, "CLIP checkpoint")
    _expect(
        clip64.get("implementation_sha256"),
        EXPECTED_CLIP_IMPLEMENTATION_SHA256,
        "CLIP implementation",
    )
    _expect(clip64.get("records"), expected["records"], "CLIP records")
    _expect(clip64.get("samples_per_condition"), 1, "CLIP draws")
    _expect(
        clip64.get("condition_summary_sha256"),
        expected["condition_summary_sha256"],
        "CLIP condition summary",
    )
    _expect(
        clip64.get("reference_waveforms_sha256"),
        expected["reference_waveforms_sha256"],
        "CLIP reference",
    )
    _expect(clip64.get("generated_waveforms_sha256"), waveform_sha, "CLIP waveform")
    _expect(
        clip64.get("inference_summary_sha256"),
        inference_hash,
        "CLIP inference binding",
    )

    hashes = {name: sha256_file(Path(path)) for name, path in paths.items()}
    hashes["inference_summary"] = inference_hash
    hashes["per_record_metrics"] = str(model_agnostic["per_record_metrics_sha256"])
    return ScoreBundle(
        model_agnostic=model_agnostic,
        morphology=morphology,
        clip64=clip64,
        inference=inference,
        paths=paths,
        hashes=hashes,
    )


def _nested(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for component in path.split("/"):
        current = current[component]
    return current


def _score_value(bundle: ScoreBundle, source: str, path: str) -> float:
    if source == "model_agnostic":
        value = _nested(bundle.model_agnostic["metrics"], path)
    elif source == "morphology":
        value = bundle.morphology["comparison"][path][
            "pooled_record_draw_median_absolute_error"
        ]
    elif source == "morphology_fixed":
        value = bundle.morphology["fixed_denominator"][path]
    elif source == "clip64":
        value = bundle.clip64["metrics"][path]
    else:
        raise ValueError(f"unknown score source {source!r}")
    return float(value)


PAIRWISE_COLUMNS = {
    "raw_mae": "mae_raw",
    "raw_mse": "mse_raw",
    "aligned_mae": "mae_aligned_centered_shared_lag",
    "aligned_mse": "mse_aligned_centered_shared_lag",
    "hr_mae_to_paired_real": "hr_error_to_paired_real",
    "qrs_f1_mean_record": "qrs_f1",
}


def _cluster_bootstrap_interval(
    differences: np.ndarray,
    patients: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    finite = np.isfinite(differences)
    unique = np.unique(patients)
    sums = np.zeros(len(unique), dtype=np.float64)
    counts = np.zeros(len(unique), dtype=np.int64)
    group = pd.Series(np.arange(len(unique)), index=unique).loc[patients].to_numpy()
    np.add.at(sums, group[finite], differences[finite])
    np.add.at(counts, group[finite], 1)
    rng = np.random.default_rng(seed)
    estimates: list[np.ndarray] = []
    for start in range(0, replicates, 500):
        count = min(500, replicates - start)
        sampled = rng.integers(0, len(unique), size=(count, len(unique)))
        numerator = sums[sampled].sum(axis=1)
        denominator = counts[sampled].sum(axis=1)
        estimates.append(
            np.divide(
                numerator,
                denominator,
                out=np.full(count, np.nan, dtype=np.float64),
                where=denominator > 0,
            )
        )
    values = np.concatenate(estimates)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("patient bootstrap produced no finite replicate")
    return {
        "estimate": float(np.mean(differences[finite])),
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
        "valid_replicates": int(len(values)),
    }


def pairwise_bootstrap(
    bundles: dict[str, ScoreBundle], *, replicates: int, seed: int
) -> dict[str, dict[str, dict[str, float | int]]]:
    frames = {
        model: pd.read_csv(bundle.model_agnostic["per_record_metrics"])
        for model, bundle in bundles.items()
    }
    identity = ["condition_position", "draw", "subject_id", "study_id"]
    reference = frames["diffusets"][identity]
    for model, frame in frames.items():
        if not reference.equals(frame[identity]):
            raise ValueError(f"per-record identity/order differs for {model}")
    patients = reference["subject_id"].astype(str).to_numpy()
    result: dict[str, dict[str, dict[str, float | int]]] = {}
    for baseline_index, baseline in enumerate(("ecgdiff", "sediff")):
        result[baseline] = {}
        for metric_index, (name, column) in enumerate(PAIRWISE_COLUMNS.items()):
            differences = (
                frames["diffusets"][column].to_numpy(np.float64)
                - frames[baseline][column].to_numpy(np.float64)
            )
            result[baseline][name] = _cluster_bootstrap_interval(
                differences,
                patients,
                replicates=replicates,
                seed=seed + baseline_index * 100 + metric_index,
            )
    return result


def _fmt(value: float, pattern: str, multiplier: float = 1.0) -> str:
    return format(value * multiplier, pattern)


def _table(
    bundles: dict[str, ScoreBundle],
    rows: Iterable[tuple[str, str, str, str, str, float]],
) -> list[str]:
    lines = [
        "| Metric | ECGDiff e16 | DiffuSETS clean e200 | SE-Diff v3 e195 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, source, path, direction, pattern, multiplier in rows:
        values = {
            model: _score_value(bundles[model], source, path) for model in MODEL_ORDER
        }
        finite = [value for value in values.values() if math.isfinite(value)]
        winner = None
        if direction == "lower":
            winner = min(finite)
        elif direction == "higher":
            winner = max(finite)
        rendered = []
        for model in MODEL_ORDER:
            value = values[model]
            text = _fmt(value, pattern, multiplier)
            if winner is not None and math.isclose(value, winner, abs_tol=1e-12):
                text = f"**{text}**"
            rendered.append(text)
        lines.append(f"| {label} | " + " | ".join(rendered) + " |")
    return lines


WAVEFORM_ROWS = (
    ("Raw paired MAE (mV)", "model_agnostic", "waveform_raw_primary/mae_raw", "lower", ".5f", 1.0),
    ("Raw paired MSE (mV^2)", "model_agnostic", "waveform_raw_primary/mse_raw", "lower", ".5f", 1.0),
    ("Raw global RMSE (mV)", "model_agnostic", "waveform_raw_primary/rmse_raw_from_global_mse", "lower", ".5f", 1.0),
    ("Raw mean-record RMSE (mV)", "model_agnostic", "waveform_raw_primary/mean_sample_rmse_raw", "lower", ".5f", 1.0),
    ("Local per-lead-PTP NRMSE", "model_agnostic", "waveform_raw_primary/mean_sample_lead_rmse_over_reference_lead_ptp", "lower", ".5f", 1.0),
    ("Raw mean per-lead Pearson r", "model_agnostic", "waveform_raw_primary/mean_sample_lead_pearson_r", "higher", ".5f", 1.0),
    ("Centered/aligned MAE, +/-1 s (mV)", "model_agnostic", "waveform_aligned_secondary/mae_aligned_centered_shared_lag", "lower", ".5f", 1.0),
    ("Centered/aligned MSE, +/-1 s (mV^2)", "model_agnostic", "waveform_aligned_secondary/mse_aligned_centered_shared_lag", "lower", ".5f", 1.0),
    ("Centered/aligned global RMSE (mV)", "model_agnostic", "waveform_aligned_secondary/rmse_aligned_from_global_mse", "lower", ".5f", 1.0),
    ("Centered/aligned mean-record RMSE (mV)", "model_agnostic", "waveform_aligned_secondary/mean_sample_rmse_aligned_centered_shared_lag", "lower", ".5f", 1.0),
    ("Centered/aligned Pearson r", "model_agnostic", "waveform_aligned_secondary/mean_sample_lead_pearson_r_aligned_centered_shared_lag", "higher", ".5f", 1.0),
    ("Alignment-boundary fraction (%)", "model_agnostic", "waveform_aligned_secondary/alignment_boundary_fraction", "lower", ".3f", 100.0),
)

PHYSIOLOGY_ROWS = (
    ("HR MAE to conditioning HR (bpm)", "model_agnostic", "physiology_and_qrs/hr_mae_to_condition_bpm", "lower", ".2f", 1.0),
    ("HR MAE to paired-real HR (bpm)", "model_agnostic", "physiology_and_qrs/hr_mae_to_paired_real_bpm", "lower", ".2f", 1.0),
    ("Within 10 bpm of conditioning HR (%)", "model_agnostic", "physiology_and_qrs/hr_within_10_bpm_to_condition_rate_all_eligible_records", "higher", ".2f", 100.0),
    ("Within 10 bpm of paired-real HR (%)", "model_agnostic", "physiology_and_qrs/hr_within_10_bpm_to_paired_real_rate_all_eligible_records", "higher", ".2f", 100.0),
    ("Beat-count MAE", "model_agnostic", "physiology_and_qrs/beat_count_mae", "lower", ".3f", 1.0),
    ("Mean-RR MAE (ms)", "model_agnostic", "physiology_and_qrs/mean_rr_mae_ms", "lower", ".2f", 1.0),
    ("RR Wasserstein (ms)", "model_agnostic", "physiology_and_qrs/rr_wasserstein_ms", "lower", ".2f", 1.0),
    ("SDNN MAE (ms)", "model_agnostic", "physiology_and_qrs/sdnn_mae_ms", "lower", ".2f", 1.0),
    ("QRS precision", "model_agnostic", "physiology_and_qrs/qrs_precision_paired_detections_global_counts", "higher", ".5f", 1.0),
    ("QRS recall", "model_agnostic", "physiology_and_qrs/qrs_recall_paired_detections_global_counts", "higher", ".5f", 1.0),
    ("QRS F1", "model_agnostic", "physiology_and_qrs/qrs_f1_paired_detections_global_counts", "higher", ".5f", 1.0),
    ("Matched-QRS timing MAE (ms)", "model_agnostic", "physiology_and_qrs/qrs_timing_mae_ms_global_matches", "lower", ".2f", 1.0),
    ("Generated QRS detection coverage", "model_agnostic", "physiology_and_qrs/generated_detection_coverage", "higher", ".3f", 1.0),
    ("Generated frontal-identity RMSE (mV)", "model_agnostic", "physical_consistency/generated_frontal_identity_rmse_mv", "lower", ".5f", 1.0),
    ("Paired absolute frontal-identity error (mV)", "model_agnostic", "physical_consistency/paired_absolute_frontal_identity_rmse_error_mv", "lower", ".5f", 1.0),
    ("Records exceeding 10 mV (%)", "model_agnostic", "physical_consistency/generated_above_10mv_fraction", "lower", ".3f", 100.0),
)

MORPHOLOGY_ROWS = (
    ("PR interval MAE (ms)", "morphology", "PR Interval", "lower", ".3f", 1.0),
    ("QRS duration MAE (ms)", "morphology", "QRS Duration", "lower", ".3f", 1.0),
    ("QT interval MAE (ms)", "morphology", "QT Interval", "lower", ".3f", 1.0),
    ("QTc Fridericia MAE (ms)", "morphology", "QTc Fridericia", "lower", ".3f", 1.0),
    ("ST at J+60 MAE (mV)", "morphology", "ST J+60", "lower", ".5f", 1.0),
    ("P-wave duration MAE (ms)", "morphology", "P-Wave Duration", "lower", ".3f", 1.0),
    ("T-wave duration MAE (ms)", "morphology", "T Duration", "lower", ".3f", 1.0),
    ("Generated all-metric coverage (%)", "morphology_fixed", "generated_all_metrics_finite_rate", "higher", ".3f", 100.0),
    ("Paired all-metric coverage (%)", "morphology_fixed", "paired_all_metrics_finite_rate", "higher", ".3f", 100.0),
)

CLIP_ROWS = (
    ("FID", "clip64", "fid", "lower", ".3f", 1.0),
    ("Real contiguous-half FID", "clip64", "real_split_fid_contiguous_halves", "none", ".3f", 1.0),
    ("Manifold precision, k=3", "clip64", "precision_k3", "higher", ".4f", 1.0),
    ("Manifold recall, k=3", "clip64", "recall_k3", "higher", ".4f", 1.0),
    ("Manifold F1, k=3", "clip64", "manifold_f1_k3", "higher", ".4f", 1.0),
    ("ECG-text cosine", "clip64", "text_cosine", "higher", ".4f", 1.0),
    ("Reference ECG-text cosine", "clip64", "reference_text_cosine", "none", ".4f", 1.0),
    ("rCLIP ratio of means", "clip64", "rclip_ratio_of_means", "higher", ".4f", 1.0),
    ("Legacy rFID, generated/real-split", "clip64", "rfid_arxiv_v1_generated_real_over_real_split", "lower", ".4f", 1.0),
    ("ICLR rFID, real-split/sum", "clip64", "rfid_iclr2026_real_split_over_sum", "higher", ".4f", 1.0),
)


def _ci_text(item: dict[str, Any], pattern: str = ".5f") -> str:
    return f"[{format(float(item['lower']), pattern)}, {format(float(item['upper']), pattern)}]"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Clean DiffuSETS full test comparison (2026-09-14)",
        "",
        "This is the registered same-scorer comparison of the local patient-disjoint ",
        "DiffuSETS seed-2026 suite against ECGDiff epoch 16 and the local SE-Diff v3 ",
        "epoch-195 lock. No author-released DiffuSETS checkpoint or saved generation ",
        "is used in any current score.",
        "",
        "## Direct result",
        "",
        "The clean DiffuSETS diffusion checkpoint performs substantially worse than both ",
        "comparators on both panels. On MIMIC, raw MAE is 0.46795 mV and QRS F1 is ",
        "0.54063, versus 0.12320/0.81272 for ECGDiff and 0.11137/0.78214 for ",
        "SE-Diff. On PTB-XL, raw MAE is 0.47626 mV and QRS F1 is 0.53613, versus ",
        "0.15051/0.77686 and 0.14366/0.75389. Clean-CLIP64 manifold recall is only ",
        "0.0037 on MIMIC and 0.0080 on PTB-XL. These independent failures support a ",
        "broad generation-quality problem. Lead-order checks pass: the saved outputs are ",
        "canonicalized before all scorers, and frontal-lead identity is evaluated after ",
        "that conversion.",
        "",
        "All entries use one draw per condition. Lower is better unless a row says ",
        "otherwise; bold marks the best literal value in a row. Aligned waveform values ",
        "are secondary diagnostics after per-lead centering and one shared circular lag ",
        "within +/-1 second.",
    ]

    for dataset, heading in (
        ("mimic", "MIMIC full shared test panel: 2,149 records / 419 patients"),
        ("ptbxl", "Fixed PTB-XL OOD panel: 1,000 records / 991 patients"),
    ):
        bundles = report["_bundles"][dataset]
        lines.extend(["", f"## {heading}", "", "### Waveform fidelity", ""])
        lines.extend(_table(bundles, WAVEFORM_ROWS))
        lines.extend(["", "### Rhythm, QRS, and physical consistency", ""])
        lines.extend(_table(bundles, PHYSIOLOGY_ROWS))
        lines.extend(["", "### ECGDeli morphology", ""])
        lines.extend(_table(bundles, MORPHOLOGY_ROWS))
        lines.extend(
            [
                "",
                "Morphology rows are median absolute paired-record errors conditional on ",
                "successful delineation; the fixed-denominator coverage rows expose every ",
                "failed record. This local canonical ECGDeli protocol is shared across the ",
                "three models and is not an author-paper evaluator reconstruction.",
                "",
                "### DiffuSETS-clean-CLIP64 seed2026 sensitivity",
                "",
            ]
        )
        lines.extend(_table(bundles, CLIP_ROWS))
        if dataset == "ptbxl":
            lines.extend(
                [
                    "",
                    "The PTB-XL feature space is strongly shifted: the real contiguous-half ",
                    "FID is about 122,317. Clean DiffuSETS therefore obtains the smallest raw ",
                    "FID while its recall is 0.008 and manifold F1 is 0.0159. Do not treat ",
                    "that FID row as evidence of useful coverage.",
                ]
            )

        lines.extend(["", "### Patient-bootstrap uncertainty", ""])
        lines.extend(
            [
                "| Model | Raw MAE 95% CI | Raw MSE 95% CI | Aligned MAE 95% CI | HR MAE to real 95% CI | Mean-record QRS F1 95% CI |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        ci_keys = (
            "waveform/mae_raw",
            "waveform/mse_raw",
            "aligned/mae",
            "physiology/hr_mae_to_paired_real",
            "qrs/f1_paired_detections",
        )
        for model in MODEL_ORDER:
            ci = bundles[model].model_agnostic[
                "confidence_intervals_95_patient_bootstrap"
            ]
            lines.append(
                f"| {MODEL_LABELS[model]} | "
                + " | ".join(_ci_text(ci[key]) for key in ci_keys)
                + " |"
            )

        lines.extend(
            [
                "",
                "The scorer intervals use 1,000 patient-clustered replicates (seed ",
                "20260903). The paired clean-minus-baseline intervals below use 10,000 ",
                "patient-clustered replicates (seed 20260914). Positive error deltas mean ",
                "clean DiffuSETS is worse; negative QRS-F1 deltas mean it is worse.",
                "",
                "| Baseline | Raw MAE delta [95% CI] | Aligned MAE delta [95% CI] | HR-to-real MAE delta [95% CI] | Mean-record QRS F1 delta [95% CI] |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        paired = report["datasets"][dataset]["pairwise_patient_bootstrap_clean_minus_baseline"]
        for baseline in ("ecgdiff", "sediff"):
            items = []
            for key, pattern in (
                ("raw_mae", ".5f"),
                ("aligned_mae", ".5f"),
                ("hr_mae_to_paired_real", ".2f"),
                ("qrs_f1_mean_record", ".5f"),
            ):
                item = paired[baseline][key]
                items.append(
                    f"{format(float(item['estimate']), pattern)} "
                    f"[{format(float(item['lower']), pattern)}, "
                    f"{format(float(item['upper']), pattern)}]"
                )
            lines.append(f"| {MODEL_LABELS[baseline]} | " + " | ".join(items) + " |")

    lines.extend(
        [
            "",
            "## Cohort and contamination boundary",
            "",
            "- MIMIC contains the complete 2,149-record native test panel selected for ",
            "  ECGDiff epoch 16. Its 419 patients map one-to-one into the clean DiffuSETS ",
            "  test role, and its text embeddings match the clean cache bit-for-bit.",
            "- PTB-XL is the fixed recovered local 1,000-record cohort (991 patients, all ",
            "  folds). It is external to the MIMIC training corpus and is not the ",
            "  authors' unreleased paper cohort. Author-package conditions are retained; ",
            "  41 records lack a finite shared HR target, so condition-HR metrics use 959.",
            "- The clean DiffuSETS VAE and U-Net were trained on its patient-disjoint ",
            "  MIMIC training role. CLIP64 was trained on train and selected on validation. ",
            "  Test records were used only here for evaluation.",
            "",
            "## Locked models and inference",
            "",
            f"- DiffuSETS suite `{ACTIVE_SUITE_ID}`: VAE `{ACTIVE_VAE_SHA256}`, CLIP64 ",
            f"  `{ACTIVE_CLIP_SHA256}`, U-Net `{ACTIVE_UNET_SHA256}`. Generation used ",
            "  epoch 200/step 54,400, base seed 20260822, 1,000-step ancestral DDPM, ",
            "  epsilon prediction, fixed-small variance, predicted-x0 clipping, and no CFG.",
            "- ECGDiff: epoch 16, step 4,624, one locked saved draw per condition.",
            "- SE-Diff: v3 seed 2026 raw epoch 195, checkpoint ",
            f"  `{EXPECTED_SEDIFF_CHECKPOINT_SHA256}`, 1,000-step ancestral DDPM and CFG 3.",
            "- DiffuSETS generation used clean Git commit ",
            "  `3e29005750b67b16a28ec770bd60ea7f11072838` with a clean worktree and ",
            "  PyTorch 2.14.0+cu126 on GPUs 2--9.",
            "",
            "## Interpretation limits",
            "",
            "- This is a fair same-cohort, same-scorer diagnostic. It does not claim the ",
            "  local baseline weights reproduce either paper's hidden evaluation protocol.",
            "- One draw cannot measure within-condition diversity. Manifold recall catches ",
            "  distributional coverage failure only through this baseline-trained feature ",
            "  space.",
            "- DiffuSETS-clean-CLIP64 is the clean reproduction's own learned evaluator, ",
            "  so its learned scores are a bespoke sensitivity analysis. Waveform, rhythm, ",
            "  QRS, physical-consistency, and ECGDeli rows are the primary independent ",
            "  evidence.",
            "- DiffuSETS decoder outputs were explicitly converted from ",
            "  `I,II,III,aVR,aVF,aVL,V1--V6` to canonical ",
            "  `I,II,III,aVR,aVL,aVF,V1--V6`. The clean evaluator converts them back ",
            "  before encoding. The prior released-output aVF/aVL ambiguity is therefore ",
            "  absent from this comparison.",
            "",
            "## Machine-readable artifacts",
            "",
            f"- Evaluation root: `{report['evaluation_root']}`",
            f"- Comparison JSON: `{report['output_json']}`",
        ]
    )
    for dataset in ("mimic", "ptbxl"):
        lines.extend(["", f"### {dataset.upper()} source hashes", ""])
        lines.extend(
            [
                "| Model | Waveforms SHA-256 | Model-agnostic report SHA-256 | Morphology report SHA-256 | Clean-CLIP64 report SHA-256 |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for model in MODEL_ORDER:
            entry = report["datasets"][dataset]["models"][model]
            lines.append(
                f"| {MODEL_LABELS[model]} | `{entry['waveforms_sha256']}` | "
                f"`{entry['report_hashes']['model_agnostic']}` | "
                f"`{entry['report_hashes']['morphology']}` | "
                f"`{entry['report_hashes']['clip64']}` |"
            )
    lines.extend(
        [
            "",
            "## Verification command",
            "",
            "```bash",
            "cd /home/nvelingker/arpa-h/diffusion/DiffuSETS_Exp",
            "/home/nvelingker/.conda/envs/ecgdiff/bin/python -m \\",
            "  paper_repro.summarize_evaluation \\",
            "  --evaluation-root \\",
            "  paper_repro/evaluation/clean_seed2026_e200_base20260822_v1 \\",
            "  --output-json \\",
            "  paper_repro/evaluation/clean_seed2026_e200_base20260822_v1/comparison.json \\",
            "  --output-markdown \\",
            "  paper_repro/evaluation/clean_seed2026_e200_base20260822_v1/comparison.md \\",
            "  --overwrite",
            "```",
            "",
            "The command re-hashes every waveform, per-record table, and morphology ",
            "artifact; checks all cohort, checkpoint, condition, reference, and evaluator ",
            "bindings; and recomputes the paired patient bootstrap before writing output.",
            "",
        ]
    )
    return "\n".join(line.rstrip() for line in lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.bootstrap_replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    root = args.evaluation_root.expanduser().resolve()
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else root / "comparison.json"
    )
    output_markdown = (
        args.output_markdown.expanduser().resolve()
        if args.output_markdown is not None
        else root / "comparison.md"
    )
    for path in (output_json, output_markdown):
        if path.exists() and not args.overwrite:
            raise FileExistsError(path)

    _require_hash(
        SCORER_ROOT / "paper_repro/v2/evaluate.py",
        EXPECTED_MODEL_AGNOSTIC_IMPLEMENTATION_SHA256,
        "model-agnostic evaluator source",
    )
    _require_hash(
        SCORER_ROOT / "paper_repro/v3/compute_paper_morphology.py",
        EXPECTED_MORPHOLOGY_IMPLEMENTATION_SHA256,
        "morphology evaluator source",
    )
    _require_hash(
        REPO_ROOT / "paper_repro/score_clip64.py",
        EXPECTED_CLIP_IMPLEMENTATION_SHA256,
        "clean CLIP64 evaluator source",
    )

    bundles: dict[str, dict[str, ScoreBundle]] = {}
    datasets: dict[str, Any] = {}
    for dataset in ("mimic", "ptbxl"):
        bundles[dataset] = {
            model: load_bundle(root, dataset, model) for model in MODEL_ORDER
        }
        paired = pairwise_bootstrap(
            bundles[dataset],
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed,
        )
        datasets[dataset] = {
            "dataset": EXPECTED[dataset]["dataset"],
            "records": EXPECTED[dataset]["records"],
            "patients": EXPECTED[dataset]["patients"],
            "condition_summary_sha256": EXPECTED[dataset][
                "condition_summary_sha256"
            ],
            "reference_waveforms_sha256": EXPECTED[dataset][
                "reference_waveforms_sha256"
            ],
            "models": {
                model: {
                    "label": MODEL_LABELS[model],
                    "waveforms": bundle.inference["waveforms"],
                    "waveforms_sha256": bundle.inference["waveforms_sha256"],
                    "inference_summary": bundle.model_agnostic["inference_summary"],
                    "inference_summary_sha256": bundle.hashes["inference_summary"],
                    "score_reports": bundle.paths,
                    "report_hashes": {
                        key: bundle.hashes[key]
                        for key in ("model_agnostic", "morphology", "clip64")
                    },
                    "model_agnostic_metrics": bundle.model_agnostic["metrics"],
                    "patient_bootstrap_95": bundle.model_agnostic[
                        "confidence_intervals_95_patient_bootstrap"
                    ],
                    "morphology": {
                        "comparison": bundle.morphology["comparison"],
                        "fixed_denominator": bundle.morphology["fixed_denominator"],
                    },
                    "diffusets_clean_clip64": bundle.clip64["metrics"],
                }
                for model, bundle in bundles[dataset].items()
            },
            "pairwise_patient_bootstrap_clean_minus_baseline": paired,
        }

    report: dict[str, Any] = {
        "schema": "diffusets_clean_three_way_evaluation_v1",
        "status": "complete",
        "evaluation_date": "2026-09-14",
        "evaluation_root": str(root),
        "output_json": str(output_json),
        "output_markdown": str(output_markdown),
        "classification": "fair same-cohort same-scorer diagnostic",
        "samples_per_condition": 1,
        "models": {
            "diffusets": {
                "suite_id": ACTIVE_SUITE_ID,
                "vae_sha256": ACTIVE_VAE_SHA256,
                "clip64_sha256": ACTIVE_CLIP_SHA256,
                "unet_sha256": ACTIVE_UNET_SHA256,
                "checkpoint_epoch": 200,
                "checkpoint_step": 54_400,
            },
            "ecgdiff": {"checkpoint": "epoch 16 / step_00004624"},
            "sediff": {
                "checkpoint": "v3 seed 2026 raw epoch 195",
                "checkpoint_sha256": EXPECTED_SEDIFF_CHECKPOINT_SHA256,
            },
        },
        "evaluator_sources": {
            "model_agnostic": {
                "path": str(SCORER_ROOT / "paper_repro/v2/evaluate.py"),
                "sha256": EXPECTED_MODEL_AGNOSTIC_IMPLEMENTATION_SHA256,
            },
            "morphology": {
                "path": str(
                    SCORER_ROOT / "paper_repro/v3/compute_paper_morphology.py"
                ),
                "sha256": EXPECTED_MORPHOLOGY_IMPLEMENTATION_SHA256,
            },
            "diffusets_clean_clip64": {
                "path": str(REPO_ROOT / "paper_repro/score_clip64.py"),
                "sha256": EXPECTED_CLIP_IMPLEMENTATION_SHA256,
                "checkpoint_sha256": ACTIVE_CLIP_SHA256,
                "independence": "bespoke baseline-trained sensitivity metric",
            },
        },
        "bootstrap": {
            "per_model_replicates": 1000,
            "per_model_seed": 20260903,
            "pairwise_replicates": args.bootstrap_replicates,
            "pairwise_seed": args.bootstrap_seed,
            "unit": "patient",
        },
        "datasets": datasets,
    }

    # Keep dataclass objects available only while rendering; JSON output remains plain.
    report["_bundles"] = bundles
    markdown = render_markdown(report)
    del report["_bundles"]
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(report, output_json)
    output_markdown.write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "complete",
                "output_json": str(output_json),
                "output_json_sha256": sha256_file(output_json),
                "output_markdown": str(output_markdown),
                "output_markdown_sha256": sha256_file(output_markdown),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
