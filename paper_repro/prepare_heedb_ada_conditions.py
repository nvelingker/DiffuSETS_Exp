"""Build DiffuSETS conditioning for selected HEEDB records.

The released DiffuSETS inference path sends only a templated diagnostic report
to ``text-embedding-ada-002``. Sex, age, and heart rate are separate scalar
conditions. This script preserves that contract while deriving HEEDB heart
rate from ECGDeli QRS detections.

Input is a CSV or Parquet table with a ``source_locator`` column containing the
zero-based HEEDB packed row and, by default, a pipe-delimited ``report``
column. Repeated rows are allowed (and expected for augmentation condition
tables). If ``--waveforms`` is supplied, the table must also contain
``waveform_index``; otherwise waveforms are read directly from the existing
HEEDB packed store.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Protocol, Sequence

import numpy as np
import pandas as pd

from paper_repro.common import (
    atomic_json_dump,
    canonical_json_bytes,
    sha256_file,
    sha256_json,
)
from utils.text_to_emb import prompt_propcess


SCHEMA = "diffusets_heedb_ada_conditions_v2"
EMBEDDING_CACHE_SCHEMA = "diffusets_exact_prompt_embedding_cache_v2"
HEEDB_REPORT_MAPPING_SCHEMA = "heedb_grouped_12sl_to_ordered_diagnoses_v1"
COLUMN_REPORT_MAPPING_SCHEMA = "pipe_delimited_condition_report_strip_fields_v1"
ADA_MODEL = "text-embedding-ada-002"
ADA_DIMENSION = 1536
ADA_MAX_INPUT_TOKENS = 8191
DEFAULT_MAX_API_PROMPTS = 256
ECGDELI_HEART_RATE_PROTOCOL = "ecgdeli_qrs_first_valid_lead_v1"
RELEASED_PROMPT_FORMATTER_SHA256 = (
    "a743e5ff36e716248f794ac27d456807e0ddde8d8561a1b8244c69b993c7a66d"
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
RELEASED_METADATA_ORDER = ("gender", "age", "heart rate")
HEEDB_REPORT_GROUP_ORDER = (
    "Rhythm",
    "Ectopy",
    "Conduction",
    "SA Node",
    "Axis/Hypertrophy",
    "Prior Infarct",
    "Ischemia/ST-T",
    "Overall",
    "Data Quality",
)
_HEEDB_REPORT_GROUP_PATTERN = re.compile(
    r"(?:^| )(" + "|".join(map(re.escape, HEEDB_REPORT_GROUP_ORDER)) + r"): "
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ECGDIFF_SOURCE = REPOSITORY_ROOT.parent / "ecgdiff/src"
DEFAULT_ECGDELI_SOURCE = REPOSITORY_ROOT.parents[1] / "src"
DEFAULT_PACKED_ROOT = Path(
    "/home/nvelingker/common-data/arpa-h/ca/derived/parcc_archive/mkeoliya/heedb_full2"
)
DEFAULT_METADATA_INDEX = Path(
    "/home/nvelingker/common-data/arpa-h/ca/derived/ecgdiff/heedb/ecgdiff_metadata_v1.npy"
)
DEFAULT_SHARED_ROOT = Path(
    "/home/nvelingker/common-data/arpa-h/diffusion/baselines/diffusets/conditioning/"
    "diffusets_heedb_ada002_v1"
)
DEFAULT_EMBEDDING_CACHE = DEFAULT_SHARED_ROOT / "cache" / ADA_MODEL


class EmbeddingsEndpoint(Protocol):
    def create(self, *, input: list[str], model: str) -> Any: ...


class _RequestsEmbeddingsEndpoint:
    """Small official-endpoint fallback for environments without the OpenAI SDK."""

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float,
        max_retries: int,
    ) -> None:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retries = Retry(
            total=max_retries,
            read=max_retries,
            connect=max_retries,
            status=max_retries,
            allowed_methods=frozenset({"POST"}),
            status_forcelist=(408, 409, 429, 500, 502, 503, 504),
            backoff_factor=1.0,
            respect_retry_after_header=True,
        )
        self._session = requests.Session()
        self._session.mount("https://", HTTPAdapter(max_retries=retries))
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._timeout_seconds = timeout_seconds

    def create(self, *, input: list[str], model: str) -> Any:
        response = self._session.post(
            "https://api.openai.com/v1/embeddings",
            headers=self._headers,
            json={"input": input, "model": model},
            timeout=self._timeout_seconds,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"OpenAI embeddings request failed with HTTP {response.status_code}"
            )
        try:
            payload = response.json()
            data = [
                SimpleNamespace(index=int(item["index"]), embedding=item["embedding"])
                for item in payload["data"]
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("OpenAI embeddings HTTP response is malformed") from error
        return SimpleNamespace(data=data)


@dataclass(frozen=True, slots=True)
class ECGDeliMeasurement:
    packed_row: int
    heart_rate_bpm: float | None
    status: str
    beat_times_seconds: tuple[float, ...]
    diagnostics: dict[str, Any]
    protocol: dict[str, Any]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingCacheStats:
    unique_prompts: int
    cache_hits: int
    cache_misses: int
    api_prompts: int
    endpoint_method: str | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create row-aligned DiffuSETS Ada-002, sex, age, and ECGDeli heart-rate "
            "conditions for a selected HEEDB table."
        )
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="CSV or Parquet condition table"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output package directory. By default, use a deterministic package under "
            f"{DEFAULT_SHARED_ROOT / 'packages'}."
        ),
    )
    parser.add_argument("--source-row-column", default="source_locator")
    parser.add_argument("--waveform-index-column", default="waveform_index")
    parser.add_argument(
        "--report-source",
        choices=("column", "packed"),
        default="column",
        help=(
            "Use the input table's pipe-delimited --report-column (default), stripping "
            "whitespace around each field to reproduce DiffuSETS' no-space delimiter; "
            "or reconstruct diagnoses from the row-aligned packed HEEDB report."
        ),
    )
    parser.add_argument("--report-column", default="report")
    parser.add_argument("--packed-root", type=Path, default=DEFAULT_PACKED_ROOT)
    parser.add_argument("--metadata-index", type=Path, default=DEFAULT_METADATA_INDEX)
    parser.add_argument(
        "--waveforms",
        type=Path,
        default=None,
        help="Optional row-aligned NPY waveform cache, for example real_waveforms.npy",
    )
    parser.add_argument(
        "--waveform-sample-rate-hz",
        type=float,
        default=None,
        help="Required with --waveforms; packed HEEDB input is read at 256 Hz otherwise",
    )
    parser.add_argument("--ecgdiff-source", type=Path, default=DEFAULT_ECGDIFF_SOURCE)
    parser.add_argument("--ecgdeli-source", type=Path, default=DEFAULT_ECGDELI_SOURCE)
    parser.add_argument(
        "--ecgdeli-workers",
        type=int,
        default=1,
        help="Parallel ECGDeli processes; each process can require substantial memory",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument(
        "--max-api-prompts",
        type=int,
        default=DEFAULT_MAX_API_PROMPTS,
        help=(
            "Refuse a run when more than this many unique cache misses would be sent "
            f"to Ada (default: {DEFAULT_MAX_API_PROMPTS})."
        ),
    )
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=DEFAULT_EMBEDDING_CACHE,
        help=(
            "Persistent content-addressed Ada cache shared across cohorts and reruns "
            f"(default: {DEFAULT_EMBEDDING_CACHE})."
        ),
    )
    parser.add_argument("--openai-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--openai-max-retries", type=int, default=5)
    parser.add_argument(
        "--api-key-environment-variable",
        default="OPENAI_API_KEY",
        help="Name of the environment variable containing the API key; the key is never saved",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Run HEEDB/ECGDeli preparation and write no embeddings or API request",
    )
    return parser.parse_args(argv)


def diffusets_prompt(report: str) -> str:
    """Apply the author-released DiffuSETS report template without normalization."""

    if not isinstance(report, str) or not report:
        raise ValueError("DiffuSETS report must be a nonempty string")
    prompt = prompt_propcess(report)
    if not prompt:
        raise ValueError("DiffuSETS report produced an empty prompt")
    # Ada-002 uses a byte-level tokenizer, so UTF-8 byte length is a
    # conservative upper bound on token count when tiktoken is unavailable.
    prompt_bytes = len(prompt.encode("utf-8"))
    if prompt_bytes > ADA_MAX_INPUT_TOKENS:
        raise ValueError(
            "DiffuSETS prompt may exceed Ada-002's input limit: "
            f"{prompt_bytes} UTF-8 bytes > {ADA_MAX_INPUT_TOKENS}"
        )
    return prompt


def validate_released_prompt_formatter() -> Path:
    """Fail if the imported formatter no longer matches the released repository blob."""

    source = REPOSITORY_ROOT / "utils/text_to_emb.py"
    actual = sha256_file(source)
    if actual != RELEASED_PROMPT_FORMATTER_SHA256:
        raise RuntimeError(
            "DiffuSETS prompt formatter differs from the released source: "
            f"expected {RELEASED_PROMPT_FORMATTER_SHA256}, found {actual}"
        )
    return source


def condition_report_to_diffusets_report(report: str) -> str:
    """Reproduce DiffuSETS' report-list join for a classification condition row.

    The classification cohorts render source 12SL fields as ``" | "`` for
    readability. The released DiffuSETS loader joined the same kind of report
    fields with a bare ``"|"``. Strip only field-edge whitespace and preserve
    field text, case, punctuation, and order.
    """

    if not isinstance(report, str) or not report:
        raise ValueError("HEEDB condition report must be a nonempty string")
    diagnoses = [diagnosis.strip() for diagnosis in report.split("|")]
    if not diagnoses or any(not diagnosis for diagnosis in diagnoses):
        raise ValueError("HEEDB condition report contains an empty diagnosis field")
    return "|".join(diagnoses)


def heedb_report_to_diffusets_report(report: str) -> str:
    """Recover ordered 12SL diagnoses and delimit them as DiffuSETS expects.

    HEEDB's packed assistant report groups source 12SL descriptions into
    category sentences. DiffuSETS expects individual ordered report strings
    separated by ``|`` before applying ``prompt_propcess``. Category labels and
    the punctuation introduced by the HEEDB packer are therefore structural,
    not diagnosis text sent to Ada.
    """

    if not isinstance(report, str) or not report:
        raise ValueError("HEEDB report must be a nonempty string")
    matches = list(_HEEDB_REPORT_GROUP_PATTERN.finditer(report))
    if not matches or matches[0].start() != 0:
        raise ValueError("HEEDB report does not start with a recognized 12SL group")
    group_positions = [
        HEEDB_REPORT_GROUP_ORDER.index(match.group(1)) for match in matches
    ]
    if len(group_positions) != len(set(group_positions)) or group_positions != sorted(
        group_positions
    ):
        raise ValueError("HEEDB report groups are duplicated or out of canonical order")

    diagnoses: list[str] = []
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(report)
        grouped_text = report[match.end() : stop]
        if not grouped_text.endswith("."):
            raise ValueError(
                f"HEEDB report group {match.group(1)!r} lacks terminal punctuation"
            )
        # format_codes() added the final period and used '; ' only to join
        # distinct 12SL descriptions within this group.
        for diagnosis in grouped_text[:-1].split("; "):
            if not diagnosis or "|" in diagnosis:
                raise ValueError("HEEDB report contains an invalid 12SL diagnosis")
            diagnoses.append(diagnosis)
    if not diagnoses:
        raise ValueError("HEEDB report contains no 12SL diagnoses")
    return "|".join(diagnoses)


def heart_rate_from_times(times_seconds: Sequence[float]) -> float:
    """Use the existing ECGDeli evaluation definition: 60 / mean synchronized RR."""

    times = np.asarray(times_seconds, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError("heart rate requires at least two finite ECGDeli R-peak times")
    intervals = np.diff(times)
    if not np.all(intervals > 0):
        raise ValueError("ECGDeli R-peak times must be strictly increasing")
    value = float(60.0 / intervals.mean())
    if not math.isfinite(value) or value <= 0:
        raise ValueError("ECGDeli produced an invalid heart rate")
    return value


def _add_import_root(path: Path) -> None:
    value = str(path.expanduser().resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def _measure_ecgdeli(
    packed_row: int,
    waveform: np.ndarray,
    sample_rate_hz: float,
    ecgdeli_source: str,
    protocol: dict[str, Any],
) -> ECGDeliMeasurement:
    """Process-safe ECGDeli QRS worker receiving one selected waveform.

    HEEDB includes records whose leads cover different time spans. A global
    synchronization vote is invalid for those records. Match DiffuSETS'
    released lead-selection policy, but replace XQRS with ECGDeli: preprocess
    once, then use the first canonical lead with at least two valid ECGDeli R
    peaks. P- and T-wave delineation is intentionally not run for heart rate.
    """

    try:
        source_root = Path(ecgdeli_source).expanduser().resolve()
        metric_root = source_root / "data_handling/metric_generation"
        _add_import_root(source_root)
        _add_import_root(metric_root)
        from data_handling.metric_generation.ecgdeli_port.preprocess import (
            preprocess_ecg,
        )
        from data_handling.metric_generation.ecgdeli_port.qrs_detection import (
            qrs_detection,
        )

        imported_sources = {
            Path(sys.modules[preprocess_ecg.__module__].__file__).resolve(),
            Path(sys.modules[qrs_detection.__module__].__file__).resolve(),
        }
        expected_sources = {
            metric_root / "ecgdeli_port/preprocess.py",
            metric_root / "ecgdeli_port/qrs_detection.py",
        }
        if imported_sources != expected_sources:
            raise RuntimeError(
                f"ECGDeli imported from {sorted(map(str, imported_sources))}, "
                f"expected {sorted(map(str, expected_sources))}"
            )

        values = np.asarray(waveform, dtype=np.float64)
        processed, processed_fs = preprocess_ecg(
            values.T, float(sample_rate_hz), target_fs=256.0
        )
        duration_seconds = values.shape[1] / float(sample_rate_hz)
        counts: dict[str, int] = {}
        selected_lead: str | None = None
        selected_times: tuple[float, ...] = ()
        for lead_index, lead_name in enumerate(CANONICAL_LEADS):
            lead = np.asarray(processed[:, lead_index], dtype=np.float64)
            if np.ptp(lead) <= 1.0e-12:
                counts[lead_name] = 0
                continue
            fpt = np.asarray(
                qrs_detection(lead, processed_fs, "mute"), dtype=np.float64
            )
            r_positions = (
                fpt[:, 5] if fpt.ndim == 2 and fpt.shape[1] >= 6 else np.empty(0)
            )
            times = r_positions / processed_fs - 1.0 / processed_fs
            valid = (
                np.isfinite(times)
                & (r_positions > 0)
                & (times >= 0)
                & (times < duration_seconds)
            )
            times = np.unique(times[valid])
            counts[lead_name] = len(times)
            if len(times) >= 2:
                selected_lead = lead_name
                selected_times = tuple(float(value) for value in times)
                break
        status = "ok" if selected_lead is not None else "insufficient_beats"
        rate = (
            heart_rate_from_times(selected_times) if selected_lead is not None else None
        )
        return ECGDeliMeasurement(
            packed_row=packed_row,
            heart_rate_bpm=rate,
            status=status,
            beat_times_seconds=selected_times,
            diagnostics={
                "selected_lead": selected_lead,
                "attempted_lead_r_peak_counts": counts,
                "processed_sample_rate_hz": float(processed_fs),
                "duration_seconds": duration_seconds,
                "confidence": None,
                "reason": None
                if selected_lead is not None
                else "no_canonical_lead_with_two_ecgdeli_r_peaks",
            },
            protocol=dict(protocol),
        )
    except (
        Exception
    ) as error:  # preserve the failing row and exception type for the audit
        return ECGDeliMeasurement(
            packed_row=packed_row,
            heart_rate_bpm=None,
            status="exception",
            beat_times_seconds=(),
            diagnostics={},
            protocol={},
            error=f"{type(error).__name__}: {error}",
        )


def measure_ecgdeli_rows(
    waveforms: dict[int, np.ndarray],
    *,
    sample_rate_hz: float,
    ecgdeli_source: Path,
    protocol: dict[str, Any],
    workers: int,
) -> dict[int, ECGDeliMeasurement]:
    if workers <= 0:
        raise ValueError("--ecgdeli-workers must be positive")
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("waveform sample rate must be finite and positive")
    arguments = [
        (
            row,
            np.ascontiguousarray(waveforms[row], dtype=np.float64),
            sample_rate_hz,
            str(ecgdeli_source.expanduser().resolve()),
            protocol,
        )
        for row in sorted(waveforms)
    ]
    measured: dict[int, ECGDeliMeasurement] = {}
    if workers == 1:
        for values in arguments:
            result = _measure_ecgdeli(*values)
            measured[result.packed_row] = result
            print(
                f"ECGDeli {len(measured)}/{len(arguments)}: packed_row={result.packed_row} "
                f"status={result.status}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = {
                pool.submit(_measure_ecgdeli, *values): values[0]
                for values in arguments
            }
            for future in as_completed(pending):
                result = future.result()
                measured[result.packed_row] = result
                print(
                    f"ECGDeli {len(measured)}/{len(arguments)}: packed_row={result.packed_row} "
                    f"status={result.status}",
                    flush=True,
                )
    failures = [
        value
        for value in measured.values()
        if value.status != "ok"
        or value.heart_rate_bpm is None
        or not math.isfinite(value.heart_rate_bpm)
    ]
    if failures:
        detail = "; ".join(
            f"row {value.packed_row}: {value.status} ({value.error or value.diagnostics})"
            for value in failures[:10]
        )
        raise RuntimeError(
            f"ECGDeli failed for {len(failures)} selected HEEDB rows: {detail}"
        )
    return measured


def _normalize_waveform(value: np.ndarray, *, label: str) -> np.ndarray:
    waveform = np.asarray(value, dtype=np.float32)
    if waveform.ndim != 2:
        raise ValueError(
            f"{label} waveform must be two dimensional, found {waveform.shape}"
        )
    if waveform.shape[0] != len(CANONICAL_LEADS) and waveform.shape[1] == len(
        CANONICAL_LEADS
    ):
        waveform = waveform.T
    if waveform.shape[0] != len(CANONICAL_LEADS) or waveform.shape[1] < 2:
        raise ValueError(
            f"{label} waveform must have shape [12,time], found {waveform.shape}"
        )
    if not np.isfinite(waveform).all():
        raise ValueError(f"{label} waveform contains nonfinite values")
    return np.ascontiguousarray(waveform)


def _read_table(path: Path) -> pd.DataFrame:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    elif suffix in {".csv", ".tsv"}:
        frame = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    else:
        raise ValueError("--input must be CSV, TSV, or Parquet")
    if frame.empty:
        raise ValueError("input condition table is empty")
    return frame.reset_index(drop=True)


def _source_rows(frame: pd.DataFrame, column: str, record_count: int) -> np.ndarray:
    if column not in frame:
        raise ValueError(f"input table lacks source-row column {column!r}")
    numeric = pd.to_numeric(frame[column], errors="raise").to_numpy(np.float64)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{column!r} must contain finite integer HEEDB packed rows")
    rows = numeric.astype(np.int64)
    if np.any(rows < 0) or np.any(rows >= record_count):
        raise ValueError(f"{column!r} contains a row outside [0,{record_count})")
    return rows


def _condition_sex(code: int) -> tuple[str, float]:
    if code == 1:
        return "M", 1.0
    if code == 2:
        return "F", 0.0
    raise ValueError(f"selected HEEDB row has unavailable or invalid sex code {code}")


def _validate_input_identity(
    frame: pd.DataFrame,
    *,
    record_ids: list[str],
    ages: np.ndarray,
    sexes: list[str],
) -> None:
    if "record_id" in frame and not np.array_equal(
        frame["record_id"].astype(str).to_numpy(), np.asarray(record_ids, dtype=str)
    ):
        raise ValueError("input record_id does not match the HEEDB packed source row")
    if "age" in frame:
        supplied = pd.to_numeric(frame["age"], errors="coerce").to_numpy(np.float64)
        finite = np.isfinite(supplied)
        if finite.any() and not np.allclose(
            supplied[finite], ages[finite], rtol=0, atol=1e-5
        ):
            raise ValueError(
                "input age differs from the authoritative HEEDB metadata index"
            )
    if "sex" in frame:
        supplied = frame["sex"].astype("string").str.strip().str.upper()
        finite = supplied.notna() & supplied.ne("")
        if finite.any() and not np.array_equal(
            supplied[finite].to_numpy(dtype=str),
            np.asarray(sexes, dtype=str)[finite.to_numpy()],
        ):
            raise ValueError(
                "input sex differs from the authoritative HEEDB metadata index"
            )


def load_heedb_conditions(
    frame: pd.DataFrame,
    *,
    source_row_column: str,
    waveform_index_column: str,
    report_source: str,
    report_column: str,
    packed_root: Path,
    metadata_index: Path,
    waveforms_path: Path | None,
    waveform_sample_rate_hz: float | None,
    ecgdiff_source: Path,
) -> tuple[pd.DataFrame, dict[int, np.ndarray], float, dict[str, Any]]:
    _add_import_root(ecgdiff_source)
    from ecgdiff.data.sources.heedb import HEEDBPackedWaveformLoader

    packed_root = packed_root.expanduser().resolve()
    metadata_index = metadata_index.expanduser().resolve()
    if not packed_root.is_dir():
        raise FileNotFoundError(packed_root)
    if not metadata_index.is_file():
        raise FileNotFoundError(metadata_index)
    metadata = np.load(metadata_index, mmap_mode="r", allow_pickle=False)
    required_fields = {"patient_id", "age_years", "sex", "institution"}
    if (
        metadata.ndim != 1
        or metadata.dtype.names is None
        or not required_fields.issubset(metadata.dtype.names)
    ):
        raise ValueError("HEEDB metadata index has an unsupported schema")
    loader = HEEDBPackedWaveformLoader(
        packed_root,
        metadata_index=metadata_index,
        target_sample_rate_hz=256,
        expected_records=len(metadata),
    )
    rows = _source_rows(frame, source_row_column, len(metadata))
    unique_rows = np.unique(rows)

    attributes: dict[int, dict[str, Any]] = {}
    for position, row_value in enumerate(unique_rows, start=1):
        row = int(row_value)
        item = metadata[row]
        age = float(item["age_years"])
        if not math.isfinite(age) or age <= 0:
            raise ValueError(f"selected HEEDB row {row} has unavailable or invalid age")
        sex, sex_numeric = _condition_sex(int(item["sex"]))
        packed_report = loader.report_text(row) if report_source == "packed" else None
        attributes[row] = {
            "record_id": loader.sample_id(row),
            "age": age,
            "sex": sex,
            "sex_numeric": sex_numeric,
            "packed_report": packed_report,
        }
        if position % 100 == 0 or position == len(unique_rows):
            print(f"HEEDB metadata/reports {position}/{len(unique_rows)}", flush=True)

    record_ids = [str(attributes[int(row)]["record_id"]) for row in rows]
    ages = np.asarray([attributes[int(row)]["age"] for row in rows], dtype=np.float64)
    sexes = [str(attributes[int(row)]["sex"]) for row in rows]
    _validate_input_identity(frame, record_ids=record_ids, ages=ages, sexes=sexes)

    source_reports: list[str] = []
    diffusets_reports: list[str] = []
    if report_source == "column":
        if report_column not in frame:
            raise ValueError(f"input table lacks report column {report_column!r}")
        for index, value in enumerate(frame[report_column]):
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"input report at row {index} must be a nonempty string"
                )
            source_reports.append(value)
            diffusets_reports.append(condition_report_to_diffusets_report(value))
    else:
        source_reports = [str(attributes[int(row)]["packed_report"]) for row in rows]
        diffusets_reports = [
            heedb_report_to_diffusets_report(report) for report in source_reports
        ]

    selected_waveforms: dict[int, np.ndarray] = {}
    waveform_source: dict[str, Any]
    if waveforms_path is not None:
        if waveform_sample_rate_hz is None:
            raise ValueError("--waveform-sample-rate-hz is required with --waveforms")
        if not math.isfinite(waveform_sample_rate_hz) or waveform_sample_rate_hz <= 0:
            raise ValueError("--waveform-sample-rate-hz must be finite and positive")
        if waveform_index_column not in frame:
            raise ValueError(
                f"input table lacks waveform-index column {waveform_index_column!r}"
            )
        waveforms_path = waveforms_path.expanduser().resolve()
        cache = np.load(waveforms_path, mmap_mode="r", allow_pickle=False)
        if cache.ndim != 3:
            raise ValueError(
                f"waveform cache must be three dimensional, found {cache.shape}"
            )
        waveform_indices = pd.to_numeric(
            frame[waveform_index_column], errors="raise"
        ).to_numpy(np.int64)
        if np.any(waveform_indices < 0) or np.any(waveform_indices >= len(cache)):
            raise ValueError("input waveform index is outside the waveform cache")
        for row_value in unique_rows:
            row = int(row_value)
            positions = np.flatnonzero(rows == row)
            indexes = np.unique(waveform_indices[positions])
            if len(indexes) != 1:
                raise ValueError(
                    f"HEEDB packed row {row} maps to multiple waveform cache rows"
                )
            selected_waveforms[row] = _normalize_waveform(
                cache[int(indexes[0])], label=f"HEEDB packed row {row}"
            )
        waveform_source = {
            "kind": "selected_npy_cache",
            "path": str(waveforms_path),
            "sample_rate_hz": float(waveform_sample_rate_hz),
        }
        sample_rate_hz = float(waveform_sample_rate_hz)
    else:
        if waveform_sample_rate_hz is not None:
            raise ValueError("--waveform-sample-rate-hz is only valid with --waveforms")
        for position, row_value in enumerate(unique_rows, start=1):
            row = int(row_value)
            selected_waveforms[row] = _normalize_waveform(
                loader.load_row(row).numpy(), label=f"HEEDB packed row {row}"
            )
            if position % 25 == 0 or position == len(unique_rows):
                print(
                    f"HEEDB packed waveforms {position}/{len(unique_rows)}", flush=True
                )
        sample_rate_hz = 256.0
        waveform_source = {
            "kind": "heedb_packed_store",
            "path": str(packed_root),
            "sample_rate_hz": sample_rate_hz,
        }

    selected_hash = hashlib.sha256()
    selected_hash.update(np.asarray(unique_rows, dtype="<i8").tobytes())
    selected_hash.update(np.asarray([sample_rate_hz], dtype="<f8").tobytes())
    for row in sorted(selected_waveforms):
        selected_hash.update(
            np.ascontiguousarray(selected_waveforms[row], dtype="<f4").tobytes()
        )
    waveform_source["selected_waveforms_sha256"] = selected_hash.hexdigest()

    output = frame.copy(deep=True)
    if "hr_bpm" in output:
        output["input_hr_bpm"] = output["hr_bpm"]
    if report_source == "packed" and report_column in output:
        output[f"input_{report_column}"] = output[report_column]
    output["condition_index"] = np.arange(len(output), dtype=np.int64)
    output["heedb_packed_row"] = rows
    output["record_id"] = record_ids
    output["report"] = source_reports
    output["diffusets_report"] = diffusets_reports
    output["age"] = ages
    output["sex"] = sexes
    output["sex_numeric_male_1_female_0"] = np.asarray(
        [attributes[int(row)]["sex_numeric"] for row in rows], dtype=np.float32
    )
    output["diffusets_prompt"] = [
        diffusets_prompt(report) for report in diffusets_reports
    ]
    output["diffusets_prompt_sha256"] = [
        hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for prompt in output["diffusets_prompt"]
    ]
    provenance = {
        "packed_root": str(packed_root),
        "packed_index": str(packed_root / "index.npz"),
        "packed_index_sha256": sha256_file(packed_root / "index.npz"),
        "metadata_index": str(metadata_index),
        "metadata_index_sha256": sha256_file(metadata_index),
        "waveforms": waveform_source,
        "report_source": report_source,
        "report_mapping": (
            HEEDB_REPORT_MAPPING_SCHEMA
            if report_source == "packed"
            else COLUMN_REPORT_MAPPING_SCHEMA
        ),
        "diffusets_report_delimiter": "|",
        "selected_source_reports_sha256": sha256_json(
            [
                [int(row), report]
                for row, report in zip(rows, source_reports, strict=True)
            ]
        ),
        "selected_diffusets_reports_sha256": sha256_json(
            [
                [int(row), report]
                for row, report in zip(rows, diffusets_reports, strict=True)
            ]
        ),
        "selected_prompts_sha256": sha256_json(
            [
                [int(row), prompt]
                for row, prompt in zip(
                    rows, output["diffusets_prompt"].tolist(), strict=True
                )
            ]
        ),
    }
    return output, selected_waveforms, sample_rate_hz, provenance


def attach_ecgdeli(
    frame: pd.DataFrame, measurements: dict[int, ECGDeliMeasurement]
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    output = frame.copy(deep=True)
    rows = output["heedb_packed_row"].to_numpy(np.int64)
    rates = np.asarray(
        [measurements[int(row)].heart_rate_bpm for row in rows], dtype=np.float64
    )
    if not np.isfinite(rates).all() or np.any(rates <= 0):
        raise ValueError("row-aligned ECGDeli heart rates are not finite and positive")
    output["hr_bpm"] = rates
    output["ecgdeli_status"] = [measurements[int(row)].status for row in rows]
    output["ecgdeli_beat_count"] = [
        len(measurements[int(row)].beat_times_seconds) for row in rows
    ]
    output["ecgdeli_r_times_seconds_json"] = [
        json.dumps(measurements[int(row)].beat_times_seconds, separators=(",", ":"))
        for row in rows
    ]
    output["ecgdeli_diagnostics_json"] = [
        json.dumps(
            measurements[int(row)].diagnostics,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for row in rows
    ]
    protocols = {
        sha256_json(value.protocol): value.protocol for value in measurements.values()
    }
    if len(protocols) != 1:
        raise ValueError(
            "selected rows were measured under different ECGDeli protocols"
        )
    protocol_sha256, protocol = next(iter(protocols.items()))
    metadata = np.column_stack(
        (
            output["sex_numeric_male_1_female_0"].to_numpy(np.float32),
            output["age"].to_numpy(np.float32),
            rates.astype(np.float32),
        )
    )
    if metadata.shape != (len(output), 3) or not np.isfinite(metadata).all():
        raise ValueError("DiffuSETS scalar metadata array is malformed")
    return (
        output,
        np.ascontiguousarray(metadata),
        {
            "protocol": protocol,
            "protocol_sha256": protocol_sha256,
            "unique_records": len(measurements),
            "heart_rate_bpm_min": float(rates.min()),
            "heart_rate_bpm_max": float(rates.max()),
            "heart_rate_bpm_mean": float(rates.mean()),
        },
    )


def _response_embeddings(response: Any, count: int) -> np.ndarray:
    try:
        indexed = [(int(item.index), item.embedding) for item in response.data]
        if sorted(index for index, _ in indexed) != list(range(count)):
            raise ValueError(
                "response indices are missing, duplicated, or out of range"
            )
        items = [embedding for _, embedding in sorted(indexed)]
        values = np.asarray(items, dtype=np.float32)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("OpenAI embeddings response is malformed") from error
    if values.shape != (count, ADA_DIMENSION) or not np.isfinite(values).all():
        raise ValueError(
            f"Ada response must contain {count} finite {ADA_DIMENSION}-dimensional embeddings"
        )
    return np.ascontiguousarray(values)


def embed_diffusets_prompts(
    prompts: Sequence[str],
    endpoint: EmbeddingsEndpoint,
    *,
    batch_size: int,
) -> tuple[np.ndarray, int]:
    """Deduplicate exact prompts, call Ada-002, and restore the original row order."""

    if batch_size <= 0:
        raise ValueError("embedding batch size must be positive")
    unique_prompts = list(dict.fromkeys(prompts))
    if not unique_prompts or any(
        not isinstance(value, str) or not value for value in unique_prompts
    ):
        raise ValueError("embedding prompts must be nonempty strings")
    unique_embeddings = np.empty((len(unique_prompts), ADA_DIMENSION), dtype=np.float32)
    for start in range(0, len(unique_prompts), batch_size):
        stop = min(start + batch_size, len(unique_prompts))
        batch = unique_prompts[start:stop]
        # This is the released API contract: templated text is the embedding
        # input, the model is text-embedding-ada-002, and dimensions are not
        # overridden. Age, sex, heart rate, and identifiers are never sent.
        response = endpoint.create(input=batch, model=ADA_MODEL)
        unique_embeddings[start:stop] = _response_embeddings(response, len(batch))
        print(f"Ada-002 prompts {stop}/{len(unique_prompts)}", flush=True)
    lookup = {prompt: index for index, prompt in enumerate(unique_prompts)}
    restored = unique_embeddings[[lookup[prompt] for prompt in prompts]]
    if restored.shape != (len(prompts), ADA_DIMENSION):
        raise AssertionError("row-aligned Ada embedding restoration failed")
    return np.ascontiguousarray(restored), len(unique_prompts)


def make_openai_endpoint(
    api_key: str,
    *,
    timeout_seconds: float,
    max_retries: int,
) -> tuple[EmbeddingsEndpoint, str]:
    if not api_key:
        raise ValueError("OpenAI API key is empty")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0 or max_retries < 0:
        raise ValueError("OpenAI timeout/retry settings are invalid")
    try:
        from openai import OpenAI
    except ImportError:
        return (
            _RequestsEmbeddingsEndpoint(
                api_key,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            ),
            "HTTPS POST https://api.openai.com/v1/embeddings",
        )
    client = OpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=max_retries)
    return client.embeddings, "OpenAI.embeddings.create"


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _embedding_cache_key(prompt: str) -> str:
    return sha256_json(
        {
            "schema": EMBEDDING_CACHE_SCHEMA,
            "model": ADA_MODEL,
            "prompt": prompt,
        }
    )


def _embedding_cache_path(cache_dir: Path, prompt: str) -> Path:
    key = _embedding_cache_key(prompt)
    return cache_dir / key[:2] / f"{key}.npz"


def _embedding_bytes_sha256(value: np.ndarray) -> str:
    canonical = np.ascontiguousarray(value, dtype="<f4")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _embedding_cache_metadata(prompt: str, value: np.ndarray) -> dict[str, Any]:
    key = _embedding_cache_key(prompt)
    return {
        "schema": EMBEDDING_CACHE_SCHEMA,
        "cache_key": key,
        "model": ADA_MODEL,
        "dimension": ADA_DIMENSION,
        "dtype": "float32",
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "embedding_sha256": _embedding_bytes_sha256(value),
    }


def _load_cached_embedding(cache_dir: Path, prompt: str) -> np.ndarray | None:
    path = _embedding_cache_path(cache_dir, prompt)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"embedding", "metadata_utf8"}:
                raise ValueError("cache archive has unexpected members")
            value = np.asarray(archive["embedding"])
            metadata_bytes = np.asarray(archive["metadata_utf8"])
            if metadata_bytes.dtype != np.uint8 or metadata_bytes.ndim != 1:
                raise ValueError("cache metadata payload is malformed")
            metadata = json.loads(metadata_bytes.tobytes().decode("utf-8"))
    except Exception as error:
        raise RuntimeError(f"Ada cache entry cannot be read: {path}") from error
    if (
        value.dtype != np.float32
        or value.shape != (ADA_DIMENSION,)
        or not np.isfinite(value).all()
    ):
        raise RuntimeError(
            f"Ada cache entry is malformed; remove it before retrying: {path}"
        )
    expected = _embedding_cache_metadata(prompt, value)
    if metadata != expected:
        raise RuntimeError(
            "Ada cache entry failed its prompt/model/content manifest; remove it before "
            f"retrying: {path}"
        )
    return np.ascontiguousarray(value)


def _store_cached_embedding(
    cache_dir: Path, prompt: str, embedding: np.ndarray
) -> None:
    value = np.asarray(embedding, dtype=np.float32)
    if value.shape != (ADA_DIMENSION,) or not np.isfinite(value).all():
        raise ValueError("refusing to cache a malformed Ada embedding")
    path = _embedding_cache_path(cache_dir, prompt)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.ascontiguousarray(value)
    metadata = canonical_json_bytes(_embedding_cache_metadata(prompt, value))
    _atomic_npz(
        path,
        embedding=value,
        metadata_utf8=np.frombuffer(metadata, dtype=np.uint8).copy(),
    )


@contextmanager
def _exclusive_embedding_cache(cache_dir: Path) -> Iterator[None]:
    """Serialize cache misses so concurrent jobs cannot bill the same prompt twice."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / ".cache.lock"
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def embed_diffusets_prompts_cached(
    prompts: Sequence[str],
    *,
    cache_dir: Path,
    endpoint_factory: Callable[[], tuple[EmbeddingsEndpoint, str]],
    batch_size: int,
    max_api_prompts: int | None = None,
) -> tuple[np.ndarray, EmbeddingCacheStats]:
    """Reuse exact Ada-002 prompt embeddings and call the API only for cache misses."""

    if batch_size <= 0:
        raise ValueError("embedding batch size must be positive")
    if max_api_prompts is not None and max_api_prompts < 0:
        raise ValueError("maximum API prompts must be nonnegative")
    unique_prompts = list(dict.fromkeys(prompts))
    if not unique_prompts or any(
        not isinstance(value, str) or not value for value in unique_prompts
    ):
        raise ValueError("embedding prompts must be nonempty strings")

    cache_dir = cache_dir.expanduser().resolve()
    embeddings_by_prompt: dict[str, np.ndarray] = {}
    endpoint_method: str | None = None
    api_prompts = 0
    with _exclusive_embedding_cache(cache_dir):
        missing: list[str] = []
        for prompt in unique_prompts:
            cached = _load_cached_embedding(cache_dir, prompt)
            if cached is None:
                missing.append(prompt)
            else:
                embeddings_by_prompt[prompt] = cached

        if missing:
            if max_api_prompts is not None and len(missing) > max_api_prompts:
                raise RuntimeError(
                    f"Ada cache has {len(missing)} unique misses, exceeding "
                    f"--max-api-prompts={max_api_prompts}; no API request was made"
                )
            endpoint, endpoint_method = endpoint_factory()
            for start in range(0, len(missing), batch_size):
                stop = min(start + batch_size, len(missing))
                batch = missing[start:stop]
                response = endpoint.create(input=batch, model=ADA_MODEL)
                values = _response_embeddings(response, len(batch))
                for prompt, value in zip(batch, values, strict=True):
                    _store_cached_embedding(cache_dir, prompt, value)
                    embeddings_by_prompt[prompt] = value
                api_prompts += len(batch)
                print(
                    f"Ada-002 cache misses embedded {stop}/{len(missing)} "
                    f"(cache hits={len(unique_prompts) - len(missing)})",
                    flush=True,
                )

    restored = np.asarray(
        [embeddings_by_prompt[prompt] for prompt in prompts], dtype=np.float32
    )
    if (
        restored.shape != (len(prompts), ADA_DIMENSION)
        or not np.isfinite(restored).all()
    ):
        raise AssertionError("row-aligned cached Ada embedding restoration failed")
    stats = EmbeddingCacheStats(
        unique_prompts=len(unique_prompts),
        cache_hits=len(unique_prompts) - len(missing),
        cache_misses=len(missing),
        api_prompts=api_prompts,
        endpoint_method=endpoint_method,
    )
    return np.ascontiguousarray(restored), stats


def inspect_embedding_cache(
    prompts: Sequence[str], *, cache_dir: Path
) -> EmbeddingCacheStats:
    """Validate cache entries and count exact hits without constructing an endpoint."""

    unique_prompts = list(dict.fromkeys(prompts))
    if not unique_prompts or any(
        not isinstance(value, str) or not value for value in unique_prompts
    ):
        raise ValueError("embedding prompts must be nonempty strings")
    hits = 0
    with _exclusive_embedding_cache(cache_dir.expanduser().resolve()):
        for prompt in unique_prompts:
            if (
                _load_cached_embedding(cache_dir.expanduser().resolve(), prompt)
                is not None
            ):
                hits += 1
    return EmbeddingCacheStats(
        unique_prompts=len(unique_prompts),
        cache_hits=hits,
        cache_misses=len(unique_prompts) - hits,
        api_prompts=0,
        endpoint_method=None,
    )


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def write_package(
    output_dir: Path,
    *,
    frame: pd.DataFrame,
    metadata: np.ndarray,
    embeddings: np.ndarray | None,
    summary: dict[str, Any],
) -> None:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; refusing to overwrite or repeat API billing: {output_dir}"
        )
    temporary = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        conditions_path = temporary / "conditions.parquet"
        metadata_path = temporary / "metadata_float32.npy"
        _atomic_parquet(conditions_path, frame)
        _atomic_npy(metadata_path, np.asarray(metadata, dtype=np.float32))
        artifacts: dict[str, dict[str, Any]] = {
            "conditions": {
                "path": "conditions.parquet",
                "sha256": sha256_file(conditions_path),
            },
            "metadata": {
                "path": "metadata_float32.npy",
                "sha256": sha256_file(metadata_path),
                "shape": list(metadata.shape),
                "dtype": "float32",
                "columns": list(RELEASED_METADATA_ORDER),
            },
        }
        if embeddings is not None:
            embeddings = np.asarray(embeddings, dtype=np.float32)
            if (
                embeddings.shape != (len(frame), ADA_DIMENSION)
                or not np.isfinite(embeddings).all()
            ):
                raise ValueError("row-aligned Ada embeddings are malformed")
            text_path = temporary / "text_embeddings_float32.npy"
            dense_path = temporary / "dense_conditioning_float32.npy"
            dense = np.ascontiguousarray(np.concatenate((embeddings, metadata), axis=1))
            _atomic_npy(text_path, embeddings)
            _atomic_npy(dense_path, dense)
            artifacts.update(
                text_embeddings={
                    "path": "text_embeddings_float32.npy",
                    "sha256": sha256_file(text_path),
                    "shape": list(embeddings.shape),
                    "dtype": "float32",
                },
                dense_conditioning={
                    "path": "dense_conditioning_float32.npy",
                    "sha256": sha256_file(dense_path),
                    "shape": list(dense.shape),
                    "dtype": "float32",
                    "columns": ["ada_002[1536]", *RELEASED_METADATA_ORDER],
                },
            )
        payload = dict(summary)
        payload["artifacts"] = artifacts
        atomic_json_dump(payload, temporary / "summary.json")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _resolved_path(path: Path | None) -> str | None:
    return None if path is None else str(path.expanduser().resolve())


def _ecgdeli_runtime_identity(
    ecgdiff_source: Path, ecgdeli_source: Path
) -> dict[str, Any]:
    """Bind the exact ECGDeli wrapper, detector source, dependencies, and settings."""

    source_root = ecgdiff_source.expanduser().resolve()
    _add_import_root(source_root)
    import ecgdiff.evaluation.beats as beats

    expected = source_root / "ecgdiff/evaluation/beats.py"
    actual = Path(beats.__file__).resolve()
    if actual != expected:
        raise RuntimeError(
            f"ecgdiff.evaluation.beats imported from {actual}, expected {expected}"
        )
    source_protocol = beats.protocol_metadata(ecgdeli_source.expanduser().resolve())
    protocol = {
        "id": ECGDELI_HEART_RATE_PROTOCOL,
        "detector": "ECGDeli_Python_qrs_detection",
        "full_pqrst_delineation": False,
        "source_available": source_protocol["source_available"],
        "source_sha256": source_protocol["source_sha256"],
        "source_files": source_protocol["source_files"],
        "dependencies": source_protocol["dependencies"],
        "input_units": "mV",
        "input_lead_order": list(CANONICAL_LEADS),
        "target_sample_rate_hz": 256.0,
        "resampling": source_protocol["resampling"],
        "preprocessing": source_protocol["preprocessing"],
        "selection": (
            "first canonical lead with at least two finite strictly ordered "
            "ECGDeli R peaks"
        ),
        "partial_or_sequential_leads_supported": True,
        "coordinate_conversion": "(one_based_r_sample - 1) / processed_sample_rate_hz",
        "heart_rate_bpm": "60 / mean(diff(r_times_seconds))",
    }
    protocol["fingerprint"] = sha256_json(protocol)
    return {
        "beats_source": str(actual),
        "beats_source_sha256": sha256_file(actual),
        "protocol": protocol,
        "protocol_sha256": sha256_json(protocol),
    }


def _job_spec(
    args: argparse.Namespace,
    input_path: Path,
    input_sha256: str,
    *,
    source_provenance: dict[str, Any],
    ecgdeli_runtime: dict[str, Any],
) -> dict[str, Any]:
    """Describe every input choice that can change the produced package."""

    return {
        "schema": SCHEMA,
        "implementation_source": str(Path(__file__).resolve()),
        "implementation_source_sha256": sha256_file(Path(__file__).resolve()),
        "input": str(input_path),
        "input_sha256": input_sha256,
        "source_row_column": args.source_row_column,
        "waveform_index_column": args.waveform_index_column,
        "report_source": args.report_source,
        "report_column": args.report_column,
        "report_mapping": (
            HEEDB_REPORT_MAPPING_SCHEMA
            if args.report_source == "packed"
            else COLUMN_REPORT_MAPPING_SCHEMA
        ),
        "prompt_formatter_sha256": RELEASED_PROMPT_FORMATTER_SHA256,
        "packed_root": _resolved_path(args.packed_root),
        "metadata_index": _resolved_path(args.metadata_index),
        "waveforms": _resolved_path(args.waveforms),
        "waveform_sample_rate_hz": args.waveform_sample_rate_hz,
        "ecgdiff_source": _resolved_path(args.ecgdiff_source),
        "ecgdeli_source": _resolved_path(args.ecgdeli_source),
        "selected_source_provenance": source_provenance,
        "ecgdeli_runtime": ecgdeli_runtime,
        "prepare_only": bool(args.prepare_only),
        "embedding_model": None if args.prepare_only else ADA_MODEL,
        "embedding_cache_schema": None if args.prepare_only else EMBEDDING_CACHE_SCHEMA,
        "embedding_cache_directory": _resolved_path(args.embedding_cache_dir),
    }


def _safe_package_label(input_path: Path) -> str:
    parts = (input_path.parent.parent.name, input_path.parent.name, input_path.stem)
    raw = "_".join(part for part in parts if part)
    label = "".join(
        character.lower() if character.isalnum() else "_" for character in raw
    )
    return "_".join(filter(None, label.split("_"))) or "heedb_conditions"


def default_output_dir(input_path: Path, job_spec_sha256: str) -> Path:
    """Return a descriptive, deterministic path in shared DiffuSETS storage."""

    return (
        DEFAULT_SHARED_ROOT
        / "packages"
        / (f"{_safe_package_label(input_path)}_{job_spec_sha256[:12]}")
    )


def _reuse_existing_package(
    output_dir: Path,
    *,
    job_spec_sha256: str,
    expected_status: str,
) -> bool:
    if not output_dir.exists():
        return False
    if not output_dir.is_dir():
        raise FileExistsError(
            f"output path exists and is not a directory: {output_dir}"
        )
    summary_path = output_dir / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"existing output package has no readable summary: {output_dir}"
        ) from error
    if (
        summary.get("schema") != SCHEMA
        or summary.get("job_spec_sha256") != job_spec_sha256
        or summary.get("status") != expected_status
    ):
        raise FileExistsError(
            "output directory belongs to a different or incomplete job; choose another "
            f"--output-dir: {output_dir}"
        )
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError(
            f"existing output package has no artifact manifest: {output_dir}"
        )
    for label, artifact in artifacts.items():
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            raise RuntimeError(f"existing output artifact {label!r} is malformed")
        path = output_dir / artifact["path"]
        if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
            raise RuntimeError(
                f"existing output artifact {label!r} failed hash validation: {path}"
            )
    return True


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    prompt_source = validate_released_prompt_formatter()
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if (
        args.ecgdeli_workers <= 0
        or args.embedding_batch_size <= 0
        or args.max_api_prompts < 0
    ):
        raise ValueError("worker/batch sizes must be positive and API cap nonnegative")
    input_sha256 = sha256_file(input_path)
    frame = _read_table(input_path)
    prepared, waveforms, sample_rate_hz, source_provenance = load_heedb_conditions(
        frame,
        source_row_column=args.source_row_column,
        waveform_index_column=args.waveform_index_column,
        report_source=args.report_source,
        report_column=args.report_column,
        packed_root=args.packed_root,
        metadata_index=args.metadata_index,
        waveforms_path=args.waveforms,
        waveform_sample_rate_hz=args.waveform_sample_rate_hz,
        ecgdiff_source=args.ecgdiff_source,
    )
    ecgdeli_runtime = _ecgdeli_runtime_identity(
        args.ecgdiff_source, args.ecgdeli_source
    )
    job_spec = _job_spec(
        args,
        input_path,
        input_sha256,
        source_provenance=source_provenance,
        ecgdeli_runtime=ecgdeli_runtime,
    )
    job_spec_sha256 = sha256_json(job_spec)
    output_dir = (
        default_output_dir(input_path, job_spec_sha256)
        if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )
    expected_status = "prepared_no_api" if args.prepare_only else "complete"
    if _reuse_existing_package(
        output_dir,
        job_spec_sha256=job_spec_sha256,
        expected_status=expected_status,
    ):
        print(f"Reusing verified output package {output_dir}", flush=True)
        return

    prompts = prepared["diffusets_prompt"].tolist()
    cache_dir = args.embedding_cache_dir.expanduser().resolve()
    cache_stats = inspect_embedding_cache(prompts, cache_dir=cache_dir)
    uses_default_shared_cache = cache_dir == DEFAULT_EMBEDDING_CACHE.resolve()
    print(
        "Ada cache preflight: "
        f"directory={cache_dir} shared_default={uses_default_shared_cache} "
        f"unique_prompts={cache_stats.unique_prompts} hits={cache_stats.cache_hits} "
        f"misses={cache_stats.cache_misses}",
        flush=True,
    )
    if not args.prepare_only and cache_stats.cache_misses > args.max_api_prompts:
        raise RuntimeError(
            f"Ada cache has {cache_stats.cache_misses} unique misses, exceeding "
            f"--max-api-prompts={args.max_api_prompts}; no API request was made"
        )

    measured = measure_ecgdeli_rows(
        waveforms,
        sample_rate_hz=sample_rate_hz,
        ecgdeli_source=args.ecgdeli_source,
        protocol=ecgdeli_runtime["protocol"],
        workers=args.ecgdeli_workers,
    )
    prepared, metadata, ecgdeli_provenance = attach_ecgdeli(prepared, measured)
    if ecgdeli_provenance["protocol_sha256"] != ecgdeli_runtime["protocol_sha256"]:
        raise RuntimeError("ECGDeli runtime changed between preflight and measurement")
    embeddings: np.ndarray | None = None
    unique_prompts = cache_stats.unique_prompts
    endpoint_method: str | None = None
    if not args.prepare_only:

        def endpoint_factory() -> tuple[EmbeddingsEndpoint, str]:
            api_key = os.environ.get(args.api_key_environment_variable, "")
            if not api_key:
                raise RuntimeError(
                    f"set {args.api_key_environment_variable}; Ada cache {cache_dir} is "
                    "missing one or more exact prompts"
                )
            return make_openai_endpoint(
                api_key,
                timeout_seconds=args.openai_timeout_seconds,
                max_retries=args.openai_max_retries,
            )

        embeddings, cache_stats = embed_diffusets_prompts_cached(
            prompts,
            cache_dir=cache_dir,
            endpoint_factory=endpoint_factory,
            batch_size=args.embedding_batch_size,
            max_api_prompts=args.max_api_prompts,
        )
        unique_prompts = cache_stats.unique_prompts
        endpoint_method = cache_stats.endpoint_method
    summary = {
        "schema": SCHEMA,
        "status": expected_status,
        "job_spec": job_spec,
        "job_spec_sha256": job_spec_sha256,
        "records": len(prepared),
        "unique_heedb_records": len(waveforms),
        "unique_prompts": unique_prompts,
        "input": str(input_path),
        "input_sha256": input_sha256,
        "source": source_provenance,
        "ecgdeli": ecgdeli_provenance,
        "text_condition": {
            "model": ADA_MODEL,
            "dimension": ADA_DIMENSION,
            "endpoint_method": endpoint_method,
            "prompt_formatter": "utils.text_to_emb.prompt_propcess",
            "prompt_formatter_source": str(prompt_source),
            "prompt_formatter_source_sha256": sha256_file(prompt_source),
            "prompt_formatter_released_sha256_verified": True,
            "report_mapping": job_spec["report_mapping"],
            "case_transform": "none",
            "deduplicated_exact_prompts": True,
            "api_fields_sent": ["input", "model"],
            "api_input_transport": "batched_array_of_exact_prompt_strings",
            "patient_metadata_sent_to_api": False,
            "persistent_cache": {
                "schema": EMBEDDING_CACHE_SCHEMA,
                "directory": str(cache_dir),
                "uses_default_shared_cache": uses_default_shared_cache,
                "key": "sha256(canonical_json({schema,model,prompt}))",
                "entry_format": "atomic_npz_with_prompt_model_content_manifest",
                "unique_prompt_hits": cache_stats.cache_hits,
                "unique_prompt_misses": cache_stats.cache_misses,
                "api_prompts": cache_stats.api_prompts,
                "would_request_prompts": cache_stats.cache_misses
                if args.prepare_only
                else cache_stats.api_prompts,
                "max_api_prompts": args.max_api_prompts,
                "exclusive_lock": True,
            },
        },
        "numeric_condition": {
            "order": list(RELEASED_METADATA_ORDER),
            "convention": "sex_M1_F0_then_age_years_then_ecgdeli_hr_bpm",
        },
    }
    write_package(
        output_dir,
        frame=prepared,
        metadata=metadata,
        embeddings=embeddings,
        summary=summary,
    )
    print(f"Wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
