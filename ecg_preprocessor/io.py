"""
Input adapters: CSV files and JSON payloads.

The acquisition device records one lead (or one 3-lead bank) at a time, then
the client combines them into a multi-lead payload. We accept both the
pre-combined CSV case and the raw per-lead JSON case.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Union

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from .constants import LEAD_INDEX, LEAD_ORDER, TARGET_FS
from .exceptions import InvalidInputError

PathLike = Union[str, Path]
JSONPayload = Mapping[str, object]


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def load_csv(
    path: PathLike,
    sample_rate: int = TARGET_FS,
    lead_columns: Iterable[str] | None = None,
) -> tuple[np.ndarray, int]:
    """Load a 12-lead (or partial) ECG from a CSV file.

    The CSV must have one column per lead with header names matching the
    canonical lead names (case-insensitive). Extra columns are ignored,
    missing leads are zero-filled (the alignment step will tolerate this
    but the SQI check will flag them).

    Parameters
    ----------
    path
        Path to the CSV.
    sample_rate
        Sampling rate the CSV was recorded at. If different from TARGET_FS
        the signal is polyphase-resampled.
    lead_columns
        Optional explicit list of column names to load. When None, all
        columns whose header matches a canonical lead name are used.

    Returns
    -------
    signal : np.ndarray, shape (n_leads_found, n_samples)
        Raw signal array, leads ordered per LEAD_ORDER (missing leads are
        zero rows).
    fs : int
        Sampling rate (equal to TARGET_FS after any resampling).
    """
    path = Path(path)
    if not path.exists():
        raise InvalidInputError(f"CSV not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise InvalidInputError(f"Failed to parse CSV: {exc}") from exc

    if df.empty:
        raise InvalidInputError("CSV is empty")

    # Normalize column names to match LEAD_ORDER casing.
    col_map = {c.strip().upper(): c for c in df.columns}
    canonical_upper = {name.upper(): name for name in LEAD_ORDER}

    if lead_columns is not None:
        wanted = {c.strip().upper() for c in lead_columns}
    else:
        wanted = set(col_map) & set(canonical_upper)

    if not wanted:
        raise InvalidInputError(
            f"No recognizable lead columns in CSV. Columns found: {list(df.columns)}. "
            f"Expected any of: {LEAD_ORDER}"
        )

    signal = np.zeros((len(LEAD_ORDER), len(df)), dtype=np.float32)
    for upper_name in wanted:
        if upper_name not in canonical_upper:
            continue
        canonical = canonical_upper[upper_name]
        idx = LEAD_INDEX[canonical]
        col = col_map[upper_name]
        series = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)
        if np.isnan(series).any():
            # Replace NaNs with zero rather than failing — SQI will catch it.
            series = np.nan_to_num(series, nan=0.0)
        signal[idx] = series

    fs = int(sample_rate)
    if fs != TARGET_FS:
        signal = _resample(signal, fs, TARGET_FS)
        fs = TARGET_FS

    return signal, fs


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def load_json(
    source: PathLike | JSONPayload | str,
) -> tuple[np.ndarray, int, str]:
    """Load a single-lead JSON payload.

    Expected shape::

        { "lead_name": "V1", "samples": [...], "sample_rate": 500 }

    Parameters
    ----------
    source
        Either a filesystem path, a raw JSON string, or a dict-like payload.

    Returns
    -------
    samples : np.ndarray, shape (n_samples,)
        Single-lead signal, resampled to TARGET_FS if needed.
    fs : int
        Sampling rate (TARGET_FS).
    lead_name : str
        Canonical lead name (e.g. "V1").
    """
    payload = _coerce_json(source)

    try:
        lead_name = str(payload["lead_name"]).strip()
        raw_samples = payload["samples"]
        sample_rate = int(payload["sample_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidInputError(
            f"JSON payload missing required fields "
            f"('lead_name', 'samples', 'sample_rate'): {exc}"
        ) from exc

    # Accept several common aliases (case, spacing).
    lead_upper = lead_name.upper().replace(" ", "")
    alias_map = {name.upper(): name for name in LEAD_ORDER}
    if lead_upper not in alias_map:
        raise InvalidInputError(
            f"Unknown lead '{lead_name}'. Expected one of: {LEAD_ORDER}"
        )
    lead_name = alias_map[lead_upper]

    samples = np.asarray(raw_samples, dtype=np.float32)
    if samples.ndim != 1:
        raise InvalidInputError(
            f"'samples' must be a 1-D array, got shape {samples.shape}"
        )
    if samples.size == 0:
        raise InvalidInputError("'samples' is empty")
    if np.isnan(samples).any():
        samples = np.nan_to_num(samples, nan=0.0)

    if sample_rate != TARGET_FS:
        samples = _resample(samples[np.newaxis, :], sample_rate, TARGET_FS)[0]

    return samples, TARGET_FS, lead_name


def combine_lead_payloads(
    payloads: Iterable[PathLike | JSONPayload | str],
    strict: bool = False,
) -> tuple[np.ndarray, int]:
    """Combine multiple single-lead JSON payloads into a (12, N) array.

    Missing leads are zero-filled. If multiple payloads target the same lead,
    the last one wins (matches "re-record this lead" UX).

    Parameters
    ----------
    payloads
        Iterable of JSON sources (path/str/dict).
    strict
        If True, raise when leads are missing. Default False: zero-fill and
        let SQI/alignment handle the downstream decision.
    """
    loaded: dict[str, np.ndarray] = {}
    max_len = 0
    for src in payloads:
        samples, _, lead_name = load_json(src)
        loaded[lead_name] = samples
        max_len = max(max_len, samples.size)

    if not loaded:
        raise InvalidInputError("No JSON payloads provided")

    missing = set(LEAD_ORDER) - set(loaded)
    if strict and missing:
        raise InvalidInputError(f"Missing leads: {sorted(missing)}")

    signal = np.zeros((len(LEAD_ORDER), max_len), dtype=np.float32)
    for lead_name, samples in loaded.items():
        idx = LEAD_INDEX[lead_name]
        # Right-pad shorter leads with zeros; alignment will shift anyway.
        signal[idx, : samples.size] = samples

    return signal, TARGET_FS


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _coerce_json(source: PathLike | JSONPayload | str) -> JSONPayload:
    """Turn the heterogeneous `source` argument into a payload dict."""
    if isinstance(source, Mapping):
        return source
    if isinstance(source, (str, Path)):
        s = str(source)
        # If it starts with '{' assume it's raw JSON, else treat as a path.
        stripped = s.lstrip()
        if stripped.startswith("{"):
            try:
                return json.loads(s)
            except json.JSONDecodeError as exc:
                raise InvalidInputError(f"Invalid JSON string: {exc}") from exc
        p = Path(s)
        if not p.exists():
            raise InvalidInputError(f"JSON file not found: {p}")
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError as exc:
            raise InvalidInputError(f"Failed to parse JSON file {p}: {exc}") from exc
    raise InvalidInputError(f"Unsupported JSON source type: {type(source).__name__}")


def _resample(signal: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """Polyphase resample each row of `signal` from fs_in to fs_out."""
    if fs_in <= 0 or fs_out <= 0:
        raise InvalidInputError("Sampling rates must be positive integers")
    if fs_in == fs_out:
        return signal
    # reduce the fraction fs_out/fs_in to small integers for resample_poly
    from math import gcd
    g = gcd(fs_in, fs_out)
    up = fs_out // g
    down = fs_in // g
    return resample_poly(signal, up, down, axis=-1).astype(np.float32)
