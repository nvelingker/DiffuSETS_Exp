from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from paper_repro.prepare_heedb_ada_conditions import (
    ADA_DIMENSION,
    ADA_MODEL,
    DEFAULT_SHARED_ROOT,
    condition_report_to_diffusets_report,
    default_output_dir,
    diffusets_prompt,
    embed_diffusets_prompts,
    embed_diffusets_prompts_cached,
    heedb_report_to_diffusets_report,
    heart_rate_from_times,
    validate_released_prompt_formatter,
)


def test_released_diffusets_prompt_is_preserved_exactly() -> None:
    report = "Sinus rhythm|PVC|Pattern of bigeminy|Abnormal ECG|"
    assert diffusets_prompt(report) == (
        "Most importantly, the 1st diagnosis is {Sinus rhythm}."
        "As a supplementary condition, the 2nd diagnosis is {PVC}."
        "As a supplementary condition, the 3rd diagnosis is {Pattern of bigeminy}."
        "As a supplementary condition, the 4th diagnosis is {Abnormal ECG}."
    )


def test_released_prompt_formatter_hash_is_pinned() -> None:
    assert validate_released_prompt_formatter().name == "text_to_emb.py"


def test_condition_report_reproduces_released_bare_pipe_join() -> None:
    source = "atrial fibrillation | with rapid ventricular response | abnormal ecg"
    report = condition_report_to_diffusets_report(source)
    assert report == "atrial fibrillation|with rapid ventricular response|abnormal ecg"
    assert diffusets_prompt(report) == (
        "Most importantly, the 1st diagnosis is {atrial fibrillation}."
        "As a supplementary condition, the 2nd diagnosis is "
        "{with rapid ventricular response}."
        "As a supplementary condition, the 3rd diagnosis is {abnormal ecg}."
    )
    with pytest.raises(ValueError, match="empty diagnosis"):
        condition_report_to_diffusets_report("sinus rhythm | | abnormal ecg")


def test_heedb_grouped_report_recovers_ordered_diffusets_diagnoses() -> None:
    source = (
        "Rhythm: sinus rhythm. "
        "Ectopy: premature atrial complexes; pattern of bigeminy. "
        "Prior Infarct: anterior infarct. "
        "Ischemia/ST-T: prolonged QTcB (>= 480 ms). "
        "Overall: abnormal electrocardiogram."
    )
    report = heedb_report_to_diffusets_report(source)
    assert report == (
        "sinus rhythm|premature atrial complexes|pattern of bigeminy|"
        "anterior infarct|prolonged QTcB (>= 480 ms)|abnormal electrocardiogram"
    )
    assert diffusets_prompt(report) == (
        "Most importantly, the 1st diagnosis is {sinus rhythm}."
        "As a supplementary condition, the 2nd diagnosis is {premature atrial complexes}."
        "As a supplementary condition, the 3rd diagnosis is {pattern of bigeminy}."
        "As a supplementary condition, the 4th diagnosis is {anterior infarct}."
        "As a supplementary condition, the 5th diagnosis is {prolonged QTcB (>= 480 ms)}."
        "As a supplementary condition, the 6th diagnosis is {abnormal electrocardiogram}."
    )


def test_ecgdeli_heart_rate_uses_mean_rr() -> None:
    assert heart_rate_from_times((0.25, 1.25, 2.25, 3.25)) == pytest.approx(60.0)
    assert heart_rate_from_times((0.0, 0.5, 1.5)) == pytest.approx(80.0)
    with pytest.raises(ValueError, match="at least two"):
        heart_rate_from_times((0.5,))
    with pytest.raises(ValueError, match="strictly increasing"):
        heart_rate_from_times((0.5, 0.5))


class _FakeEmbeddingsEndpoint:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, *, input: list[str], model: str) -> SimpleNamespace:
        self.calls.append({"input": list(input), "model": model})
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    index=index, embedding=np.full(ADA_DIMENSION, index + 1.0)
                )
                for index in reversed(range(len(input)))
            ]
        )


def test_ada_call_uses_only_exact_prompts_and_restores_duplicate_rows() -> None:
    endpoint = _FakeEmbeddingsEndpoint()
    values, unique = embed_diffusets_prompts(
        ["prompt A", "prompt A", "prompt B"], endpoint, batch_size=10
    )
    assert endpoint.calls == [{"input": ["prompt A", "prompt B"], "model": ADA_MODEL}]
    assert unique == 2
    assert values.shape == (3, ADA_DIMENSION)
    np.testing.assert_array_equal(values[0], values[1])
    assert np.all(values[0] == 1.0)
    assert np.all(values[2] == 2.0)


def test_shared_ada_cache_avoids_repeat_api_calls(tmp_path) -> None:
    endpoint = _FakeEmbeddingsEndpoint()
    first, first_stats = embed_diffusets_prompts_cached(
        ["prompt A", "prompt A", "prompt B"],
        cache_dir=tmp_path / "cache",
        endpoint_factory=lambda: (endpoint, "fake endpoint"),
        batch_size=10,
    )
    assert first_stats.cache_hits == 0
    assert first_stats.cache_misses == 2
    assert first_stats.api_prompts == 2
    assert endpoint.calls == [{"input": ["prompt A", "prompt B"], "model": ADA_MODEL}]

    def fail_if_called():
        pytest.fail("endpoint factory must not be called on an all-cache-hit rerun")

    second, second_stats = embed_diffusets_prompts_cached(
        ["prompt A", "prompt A", "prompt B"],
        cache_dir=tmp_path / "cache",
        endpoint_factory=fail_if_called,
        batch_size=1,
    )
    np.testing.assert_array_equal(second, first)
    assert second_stats.cache_hits == 2
    assert second_stats.cache_misses == 0
    assert second_stats.api_prompts == 0
    assert second_stats.endpoint_method is None

    entries = list((tmp_path / "cache").rglob("*.npz"))
    assert len(entries) == 2
    with np.load(entries[0], allow_pickle=False) as archive:
        assert set(archive.files) == {"embedding", "metadata_utf8"}


def test_cache_manifest_detects_valid_shape_content_change(tmp_path) -> None:
    endpoint = _FakeEmbeddingsEndpoint()
    embed_diffusets_prompts_cached(
        ["prompt A"],
        cache_dir=tmp_path / "cache",
        endpoint_factory=lambda: (endpoint, "fake endpoint"),
        batch_size=10,
    )
    entry = next((tmp_path / "cache").rglob("*.npz"))
    with np.load(entry, allow_pickle=False) as archive:
        metadata = archive["metadata_utf8"].copy()
    with entry.open("wb") as handle:
        np.savez(
            handle,
            embedding=np.full(ADA_DIMENSION, 99.0, dtype=np.float32),
            metadata_utf8=metadata,
        )

    with pytest.raises(RuntimeError, match="content manifest"):
        embed_diffusets_prompts_cached(
            ["prompt A"],
            cache_dir=tmp_path / "cache",
            endpoint_factory=lambda: pytest.fail("corrupt cache must fail closed"),
            batch_size=10,
        )


def test_cache_miss_cap_blocks_endpoint_construction(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="no API request was made"):
        embed_diffusets_prompts_cached(
            ["prompt A", "prompt B"],
            cache_dir=tmp_path / "cache",
            endpoint_factory=lambda: pytest.fail(
                "cap must run before endpoint construction"
            ),
            batch_size=10,
            max_api_prompts=1,
        )


class _DuplicateIndexEndpoint:
    def create(self, *, input: list[str], model: str) -> SimpleNamespace:
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=0, embedding=np.zeros(ADA_DIMENSION)),
                SimpleNamespace(index=0, embedding=np.ones(ADA_DIMENSION)),
            ]
        )


def test_ada_response_requires_exact_unique_indices() -> None:
    with pytest.raises(ValueError, match="malformed"):
        embed_diffusets_prompts(
            ["prompt A", "prompt B"], _DuplicateIndexEndpoint(), batch_size=10
        )


def test_default_output_is_deterministic_and_shared(tmp_path) -> None:
    input_path = tmp_path / "heedb" / "bigeminy" / "eval.parquet"
    output = default_output_dir(input_path, "a" * 64)
    assert output.parent == DEFAULT_SHARED_ROOT / "packages"
    assert output.name == "heedb_bigeminy_eval_aaaaaaaaaaaa"
