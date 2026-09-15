from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from paper_repro.common import sha256_file
from paper_repro.evaluation import (
    CANONICAL_FROM_DIFFUSETS,
    CANONICAL_LEADS,
    DIFFUSETS_FROM_CANONICAL,
    DIFFUSETS_LEADS,
    canonicalize_decoded,
    diffusets_order_from_canonical,
    load_condition_envelope,
)
from paper_repro.infer import main as infer_main
from paper_repro.infer import make_scheduler, sample_latents
from paper_repro.score_clip64 import fid_score


def test_lead_adapter_is_named_round_trip() -> None:
    assert CANONICAL_FROM_DIFFUSETS == (0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11)
    assert DIFFUSETS_FROM_CANONICAL == CANONICAL_FROM_DIFFUSETS
    decoded = torch.arange(2 * 1024 * 12, dtype=torch.float32).reshape(2, 1024, 12)
    canonical = canonicalize_decoded(decoded)
    assert canonical.shape == (2, 12, 1024)
    assert tuple(CANONICAL_LEADS) == tuple(
        DIFFUSETS_LEADS[index] for index in CANONICAL_FROM_DIFFUSETS
    )
    assert torch.equal(diffusets_order_from_canonical(canonical), decoded)


class _ZeroDenoiser(torch.nn.Module):
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        text: torch.Tensor,
        metadata: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        del timestep, text, metadata
        return torch.zeros_like(sample)


def _sample(seeds: tuple[int, ...]) -> torch.Tensor:
    count = len(seeds)
    device = torch.device("cpu")
    scheduler = make_scheduler(
        num_train_steps=8,
        beta_start=0.00085,
        beta_end=0.012,
        inference_steps=8,
        device=device,
    )
    return sample_latents(
        _ZeroDenoiser(),
        text=torch.zeros(count, 1, 1536),
        metadata={
            name: torch.zeros(count, 1, 1) for name in ("gender", "age", "heart rate")
        },
        seeds=seeds,
        scheduler=scheduler,
        device=device,
    )


def test_per_record_generators_are_batch_partition_invariant() -> None:
    together = _sample((20260822, 20260823, 20260824))
    separate = torch.cat(
        (_sample((20260822,)), _sample((20260823,)), _sample((20260824,)))
    )
    assert torch.equal(together, separate)


def test_sampler_rejects_condition_shape_drift() -> None:
    scheduler = make_scheduler(
        num_train_steps=4,
        beta_start=0.00085,
        beta_end=0.012,
        inference_steps=4,
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="text input"):
        sample_latents(
            _ZeroDenoiser(),
            text=torch.zeros(1, 1536),
            metadata={
                name: torch.zeros(1, 1, 1) for name in ("gender", "age", "heart rate")
            },
            seeds=(1,),
            scheduler=scheduler,
            device=torch.device("cpu"),
        )


def test_non_registered_config_is_rejected(tmp_path: Path) -> None:
    quarantined_config = (
        Path(__file__).resolve().parents[1] / "config/patient_disjoint_fsdp2.json"
    )
    with pytest.raises(ValueError, match="not the registered clean-suite config"):
        infer_main(
            [
                "--config",
                str(quarantined_config),
                "--condition-dir",
                str(tmp_path / "conditions"),
                "--output-dir",
                str(tmp_path / "output"),
                "--condition-source",
                "clean-mimic",
            ]
        )


def test_condition_envelope_checks_hashes_and_order(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv.gz"
    pd.DataFrame(
        {
            "condition_idx": [0, 1],
            "study_id": [101, 102],
            "patient_id": [7, 8],
        }
    ).to_csv(manifest, index=False, compression="gzip")
    reference = tmp_path / "reference.npy"
    np.save(reference, np.zeros((2, 12, 1024), dtype=np.float32))
    summary = {
        "schema": "sediff_reconstruction_v2",
        "stage": "evaluation_condition_package",
        "status": "frozen",
        "dataset": "fixture",
        "records": 2,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "reference_ecg": str(reference),
        "reference_ecg_sha256": sha256_file(reference),
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    loaded, _, _, frame = load_condition_envelope(tmp_path)
    assert loaded["records"] == 2
    assert frame["study_id"].tolist() == [101, 102]
    frame.loc[::-1].to_csv(manifest, index=False, compression="gzip")
    summary["manifest_sha256"] = sha256_file(manifest)
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="condition_idx"):
        load_condition_envelope(tmp_path)


def test_fid_is_zero_for_identical_features() -> None:
    generator = torch.Generator().manual_seed(2026)
    features = torch.randn(32, 64, generator=generator)
    # scipy.sqrtm can leave a tiny signed roundoff residual for a rank-deficient
    # 64-D covariance estimated from 32 rows.
    assert fid_score(features, features) == pytest.approx(0.0, abs=2e-6)
