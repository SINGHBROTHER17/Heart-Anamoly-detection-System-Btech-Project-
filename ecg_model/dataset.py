"""
dataset.py — PTB-XL Dataset Loader
====================================

Downloads, parses, and serves PTB-XL ECG records as PyTorch tensors.

PTB-XL ships 21,799 12-lead ECGs at 500 Hz. Each record comes with a set of
SCP (Standard Communications Protocol) diagnostic codes, each with a
likelihood percentage. We map a subset of codes to our 10 binary conditions
and build a multi-hot label vector.

SCP → condition mapping
-----------------------
The mapping is based on the official PTB-XL paper (Wagner et al. 2020) and
the PhysioNet SCP code definitions. Where a condition maps to multiple SCP
codes, a record is positive if ANY of the mapped codes has likelihood ≥ 50%.

Augmentations
-------------
Four augmentations are applied randomly during training:
  1. Gaussian noise injection (σ = 0.01–0.05 × signal range)
  2. Random lead dropout (1–2 leads zeroed; model must learn to work without
     a lead, matching real-world electrode-off scenarios)
  3. Time warping (±5% local stretching via linear interpolation)
  4. Amplitude scaling (×0.8–1.2 per lead)
  5. Synthetic desynchronization (±200 ms shift per lead) — simulates the
     sequential recording artifact the preprocessing pipeline corrects.
     Including this during training makes the model robust to residual
     alignment errors that slip through the SQI gate.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import wfdb
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from .model import CONDITION_NAMES, N_CONDITIONS


# ---------------------------------------------------------------------------
# SCP code → condition index mapping
# ---------------------------------------------------------------------------

# Format: condition_index -> frozenset of SCP codes that vote for it.
# We use a broad mapping to maximise training signal.

SCP_MAP: dict[int, frozenset[str]] = {
    0: frozenset({"NORM"}),                            # Normal Sinus Rhythm
    1: frozenset({"AFIB", "AFLT"}),                    # Atrial Fibrillation
    2: frozenset({"AMI", "IMI", "LMI", "ASMI",        # ST Elevation / STEMI
                  "ILMI", "IPLMI", "IPMI", "LFMI",
                  "PMI", "ALMI", "INJAS", "INJAL",
                  "INJIN", "INJLA", "STD_", "STE_"}),
    3: frozenset({"LBBB", "CLBBB"}),                   # LBBB
    4: frozenset({"RBBB", "CRBBB", "IRBBB"}),          # RBBB
    5: frozenset({"LVH"}),                             # LVH
    6: frozenset({"SBRAD", "BRAD"}),                   # Bradycardia
    7: frozenset({"STACH", "SVTACH", "PSVT"}),         # Tachycardia
    8: frozenset({"1AVB"}),                            # First Degree AV Block
    9: frozenset({"PVC", "VPCS"}),                     # PVC
}


def scp_to_labels(
    scp_codes: dict[str, float],
    threshold: float = 50.0,
) -> np.ndarray:
    """Convert a PTB-XL scp_codes dict to a multi-hot (N_CONDITIONS,) array.

    A label is positive when at least one of its mapped SCP codes has
    likelihood ≥ threshold.
    """
    labels = np.zeros(N_CONDITIONS, dtype=np.float32)
    for cond_idx, codes in SCP_MAP.items():
        for code in codes:
            if scp_codes.get(code, 0.0) >= threshold:
                labels[cond_idx] = 1.0
                break
    return labels


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class PTBXLDataset(Dataset):
    """
    Parameters
    ----------
    records_df : DataFrame
        Subset of the PTB-XL database CSV (already split into train/val/test).
    data_dir : Path
        Root of the PTB-XL download (contains the 'records500/' subdirectory).
    augment : bool
        Whether to apply training-time augmentation.
    fs : int
        Sampling rate to use. PTB-XL provides 100 Hz and 500 Hz.
    """

    def __init__(
        self,
        records_df: pd.DataFrame,
        data_dir: Path,
        augment: bool = False,
        fs: int = 500,
    ):
        self.records = records_df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.fs = fs
        self.subdir = "records500" if fs == 500 else "records100"

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.records.iloc[idx]

        # Load waveform via wfdb.
        filename = row["filename_hr"] if self.fs == 500 else row["filename_lr"]
        filepath = str(self.data_dir / filename)
        record = wfdb.rdrecord(filepath)
        signal = record.p_signal.T.astype(np.float32)  # (12, N)

        # Parse labels.
        scp = ast.literal_eval(row["scp_codes"]) if isinstance(row["scp_codes"], str) else row["scp_codes"]
        labels = scp_to_labels(scp)

        # Augmentation (training only).
        if self.augment:
            signal = augment_signal(signal, fs=self.fs)

        # Ensure exactly (12, 5000).
        signal = _ensure_shape(signal, target_len=5000)

        # Per-lead z-score (same as preprocessing pipeline).
        signal = _zscore(signal)

        return torch.from_numpy(signal), torch.from_numpy(labels)

    @property
    def label_matrix(self) -> np.ndarray:
        """(N, N_CONDITIONS) array of labels for the full dataset — used for class weighting."""
        import ast
        labels = []
        for _, row in self.records.iterrows():
            scp = ast.literal_eval(row["scp_codes"]) if isinstance(row["scp_codes"], str) else row["scp_codes"]
            labels.append(scp_to_labels(scp))
        return np.stack(labels)


# ---------------------------------------------------------------------------
# Data loading entry point
# ---------------------------------------------------------------------------

def load_ptbxl(
    data_dir: str | Path,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    fs: int = 500,
) -> tuple[PTBXLDataset, PTBXLDataset, PTBXLDataset]:
    """Download (if needed) and return train / val / test datasets.

    PTB-XL provides an official strat_fold column (1–10); we use folds 1–7
    for training, 8 for validation, 9–10 for test to match community convention.
    The caller can override with val_frac/test_frac for a random split instead,
    but the strat_fold approach is preferred for comparability.
    """
    data_dir = Path(data_dir)
    metadata_path = data_dir / "ptbxl_database.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"PTB-XL metadata not found at {metadata_path}. "
            "Run: wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/ "
            "or use the Colab cell that calls _download_ptbxl()."
        )

    df = pd.read_csv(metadata_path, index_col="ecg_id")

    # PTB-XL strat_fold: 1-7 train, 8 val, 9-10 test.
    if "strat_fold" in df.columns:
        train_df = df[df.strat_fold <= 7]
        val_df   = df[df.strat_fold == 8]
        test_df  = df[df.strat_fold >= 9]
    else:
        # Fallback to random split (for synthetic/partial downloads).
        train_df, tmp = train_test_split(df, test_size=val_frac + test_frac,
                                         random_state=seed)
        rel_val = val_frac / (val_frac + test_frac)
        val_df, test_df = train_test_split(tmp, test_size=1 - rel_val,
                                           random_state=seed)

    train_ds = PTBXLDataset(train_df, data_dir, augment=True,  fs=fs)
    val_ds   = PTBXLDataset(val_df,   data_dir, augment=False, fs=fs)
    test_ds  = PTBXLDataset(test_df,  data_dir, augment=False, fs=fs)
    return train_ds, val_ds, test_ds


def make_weighted_sampler(dataset: PTBXLDataset) -> WeightedRandomSampler:
    """Over-sample rare conditions to counter class imbalance.

    Each sample's weight is the reciprocal of the sum of its positive-label
    class frequencies. This gives rare multi-label combinations more
    representation without completely discarding the majority class.
    """
    labels = dataset.label_matrix          # (N, 10)
    # Class frequency per condition.
    freq = labels.mean(axis=0) + 1e-6      # (10,)
    # Per-sample weight: inverse sum of positive class frequencies.
    sample_weights = np.zeros(len(labels))
    for i, row in enumerate(labels):
        if row.sum() == 0:
            # "Abnormal but not in our 10 classes" — weight like the rarest class.
            sample_weights[i] = 1.0 / freq.min()
        else:
            sample_weights[i] = 1.0 / (freq * row).sum()
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True,
    )


def make_dataloaders(
    train_ds: PTBXLDataset,
    val_ds: PTBXLDataset,
    test_ds: PTBXLDataset,
    batch_size: int = 64,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    sampler = make_weighted_sampler(train_ds) if use_weighted_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

def augment_signal(signal: np.ndarray, fs: int = 500) -> np.ndarray:
    """Apply a random subset of training augmentations.

    Input/output: (12, N) float32. Each augmentation is applied with
    independent 50% probability, except desync which is applied with 70%
    probability (it's the primary artifact we want robustness to).
    """
    rng = np.random.default_rng()

    if rng.random() < 0.5:
        signal = _add_gaussian_noise(signal, rng)
    if rng.random() < 0.5:
        signal = _random_lead_dropout(signal, rng)
    if rng.random() < 0.4:
        signal = _time_warp(signal, rng, fs=fs)
    if rng.random() < 0.5:
        signal = _amplitude_scale(signal, rng)
    if rng.random() < 0.7:
        signal = _synthetic_desync(signal, rng, fs=fs, max_shift_ms=200.0)

    return signal


def _add_gaussian_noise(sig: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """σ drawn uniformly from [0.01, 0.05] × signal std."""
    std = sig.std(axis=-1, keepdims=True).clip(min=1e-6)
    sigma = rng.uniform(0.01, 0.05)
    return sig + (sigma * std * rng.standard_normal(sig.shape)).astype(np.float32)


def _random_lead_dropout(sig: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Zero out 1 or 2 randomly chosen leads."""
    n_drop = rng.integers(1, 3)
    leads = rng.choice(sig.shape[0], size=n_drop, replace=False)
    sig = sig.copy()
    sig[leads] = 0.0
    return sig


def _time_warp(sig: np.ndarray, rng: np.random.Generator, fs: int) -> np.ndarray:
    """Apply mild local time warping (±5% stretch) to the whole recording."""
    N = sig.shape[-1]
    # Random warp ratio per lead: stretch or compress up to 5%.
    warp = rng.uniform(0.95, 1.05)
    new_len = int(N * warp)
    # Resample via linear interpolation.
    old_idx = np.linspace(0, N - 1, new_len)
    new_idx = np.arange(N)
    warped = np.stack([
        np.interp(new_idx, old_idx, sig[i]) for i in range(sig.shape[0])
    ], axis=0)
    return warped.astype(np.float32)


def _amplitude_scale(sig: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Scale each lead amplitude by a per-lead factor in [0.8, 1.2]."""
    factors = rng.uniform(0.8, 1.2, size=(sig.shape[0], 1)).astype(np.float32)
    return sig * factors


def _synthetic_desync(
    sig: np.ndarray,
    rng: np.random.Generator,
    fs: int,
    max_shift_ms: float = 200.0,
) -> np.ndarray:
    """Shift individual leads by ±max_shift_ms to simulate sequential recording."""
    max_shift = int(round(max_shift_ms / 1000.0 * fs))
    out = sig.copy()
    for i in range(sig.shape[0]):
        shift = int(rng.integers(-max_shift, max_shift + 1))
        if shift == 0:
            continue
        out[i] = np.roll(sig[i], shift)
    return out


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _ensure_shape(signal: np.ndarray, target_len: int = 5000) -> np.ndarray:
    """Crop to center or zero-pad to reach exactly target_len."""
    n = signal.shape[-1]
    if n == target_len:
        return signal
    if n > target_len:
        start = (n - target_len) // 2
        return signal[:, start: start + target_len]
    # Pad
    pad = target_len - n
    left = pad // 2
    right = pad - left
    return np.pad(signal, ((0, 0), (left, right)), mode="constant")


def _zscore(signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = signal.mean(axis=-1, keepdims=True)
    std = signal.std(axis=-1, keepdims=True)
    safe_std = np.where(std < eps, 1.0, std)
    z = np.where(std < eps, 0.0, (signal - mean) / safe_std)
    return z.astype(np.float32)


# ---------------------------------------------------------------------------
# PTB-XL downloader (called from the Colab notebook)
# ---------------------------------------------------------------------------

def download_ptbxl(target_dir: str | Path = "./ptbxl") -> Path:
    """Download PTB-XL from PhysioNet using wget (available on Colab)."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    url = "https://physionet.org/files/ptb-xl/1.0.3/"
    print(f"Downloading PTB-XL to {target_dir} (~3 GB, this will take a while)...")
    os.system(
        f"wget -r -N -c -np --no-check-certificate -q "
        f"--directory-prefix={target_dir} {url} 2>&1 | tail -5"
    )
    # PhysioNet creates a subdirectory with the full URL path; find the root.
    nested = target_dir / "physionet.org" / "files" / "ptb-xl" / "1.0.3"
    if nested.exists():
        return nested
    return target_dir
