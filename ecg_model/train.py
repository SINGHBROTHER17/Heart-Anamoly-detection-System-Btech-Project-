"""
train.py — Training Loop
=========================

Trains the ECGAnomalyDetector on PTB-XL with:
  - Binary cross-entropy loss with class weighting
  - AdamW + cosine annealing LR scheduler
  - Early stopping on macro-average val AUC-ROC (patience=7)
  - Per-class AUC-ROC, F1, precision, recall logged every epoch
  - Best checkpoint saved to Drive at every validation improvement
  - Training curve and confusion matrix plotted at end

Can be called from the Colab notebook or as a standalone script.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from .model import (
    CONDITION_NAMES,
    ECGAnomalyDetector,
    ModelConfig,
    N_CONDITIONS,
    TemperatureScaler,
    build_model,
)
from .dataset import (
    PTBXLDataset,
    load_ptbxl,
    make_dataloaders,
)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Paths
    data_dir: str = "./ptbxl"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # Training
    epochs: int = 50
    batch_size: int = 64
    num_workers: int = 2

    # Optimizer
    lr: float = 3e-4
    weight_decay: float = 1e-4

    # LR schedule — cosine annealing
    lr_min: float = 1e-6
    warmup_epochs: int = 3

    # Early stopping
    patience: int = 7
    min_delta: float = 1e-4

    # Loss
    pos_weight_cap: float = 20.0   # cap to avoid huge gradients on tiny classes

    # Model
    model_cfg: Optional[ModelConfig] = None

    # Reproducibility
    seed: int = 42


# ---------------------------------------------------------------------------
# Loss with class weighting
# ---------------------------------------------------------------------------

def build_pos_weight(
    train_dataset: PTBXLDataset,
    cap: float = 20.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Compute per-class positive weight = (n_neg / n_pos), capped.

    BCEWithLogitsLoss accepts pos_weight to up-weight the minority class
    in each binary head. We cap it so extremely rare conditions don't
    dominate gradients and destabilise training.
    """
    labels = train_dataset.label_matrix       # (N, 10)
    n = len(labels)
    pos = labels.sum(axis=0).clip(min=1)      # (10,)
    neg = n - pos
    weights = (neg / pos).clip(max=cap)
    return torch.from_numpy(weights.astype(np.float32)).to(device)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_metrics(
    all_logits: np.ndarray,   # (N, 10)
    all_labels: np.ndarray,   # (N, 10)
    threshold: float = 0.5,
) -> dict[str, float | dict]:
    """Compute per-class and macro metrics from raw logits."""
    probs = 1.0 / (1.0 + np.exp(-all_logits))   # sigmoid

    # AUC-ROC — per class (skip classes with no positive examples)
    auc_per_class = {}
    for i, cond in enumerate(CONDITION_NAMES):
        if all_labels[:, i].sum() == 0 or all_labels[:, i].sum() == len(all_labels):
            auc_per_class[cond] = float("nan")
        else:
            try:
                auc_per_class[cond] = roc_auc_score(all_labels[:, i], probs[:, i])
            except Exception:
                auc_per_class[cond] = float("nan")

    valid_aucs = [v for v in auc_per_class.values() if not np.isnan(v)]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0

    # Threshold predictions
    preds = (probs >= threshold).astype(int)

    return {
        "macro_auc": macro_auc,
        "auc_per_class": auc_per_class,
        "macro_f1": f1_score(all_labels, preds, average="macro", zero_division=0),
        "macro_precision": precision_score(all_labels, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(all_labels, preds, average="macro", zero_division=0),
    }


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model: ECGAnomalyDetector,
    loader: DataLoader,
    criterion: nn.BCEWithLogitsLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    is_train: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run one pass. Returns (mean_loss, all_logits, all_labels)."""
    model.train(is_train)
    total_loss = 0.0
    all_logits, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for signals, labels in loader:
            signals = signals.to(device, non_blocking=True)
            labels  = labels.to(device,  non_blocking=True)

            logits = model(signals)
            loss = criterion(logits, labels)

            if is_train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Gradient clipping: ECG models can see large gradient spikes
                # when a batch is dominated by a single rare class.
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(signals)
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    mean_loss = total_loss / max(len(all_labels), 1)
    return mean_loss, all_logits, all_labels


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig | None = None) -> ECGAnomalyDetector:
    if cfg is None:
        cfg = TrainConfig()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)

    # --- Data ---
    print("Loading PTB-XL...")
    train_ds, val_ds, test_ds = load_ptbxl(cfg.data_dir)
    train_loader, val_loader, test_loader = make_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # --- Model ---
    model = build_model(cfg.model_cfg, device=str(device))
    counts = model.parameter_count()
    print(f"Model: {counts['trainable']:,} trainable params")

    # --- Loss ---
    pos_weight = build_pos_weight(train_ds, cap=cfg.pos_weight_cap, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Optimizer + LR schedule ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    # Linear warmup for warmup_epochs, then cosine decay to lr_min.
    def lr_lambda(epoch: int) -> float:
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / cfg.warmup_epochs
        progress = (epoch - cfg.warmup_epochs) / max(cfg.epochs - cfg.warmup_epochs, 1)
        return cfg.lr_min / cfg.lr + 0.5 * (1 - cfg.lr_min / cfg.lr) * (
            1 + np.cos(np.pi * progress)
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Training loop ---
    best_val_auc = -1.0
    patience_counter = 0
    history: list[dict] = []
    best_ckpt_path = Path(cfg.checkpoint_dir) / "best_model.pt"

    print("\nEpoch  Train-Loss  Val-Loss   Val-AUC   LR")
    print("-" * 55)

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True
        )
        val_loss, val_logits, val_labels = run_epoch(
            model, val_loader, criterion, None, device, is_train=False
        )
        scheduler.step()

        metrics = compute_metrics(val_logits, val_labels)
        val_auc = metrics["macro_auc"]
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_f1": metrics["macro_f1"],
            "lr": current_lr,
            **{f"auc_{c}": v for c, v in metrics["auc_per_class"].items()},
        }
        history.append(row)

        elapsed = time.time() - t0
        print(
            f"{epoch:5d}  {train_loss:.4f}      {val_loss:.4f}     "
            f"{val_auc:.4f}    {current_lr:.2e}   [{elapsed:.0f}s]"
        )

        # Per-class detail every 5 epochs.
        if epoch % 5 == 0:
            print("        Per-class AUC:")
            for cond, auc in metrics["auc_per_class"].items():
                print(f"          {cond:<40s}: {auc:.3f}")

        # Checkpoint on improvement.
        if val_auc > best_val_auc + cfg.min_delta:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": asdict(cfg.model_cfg) if cfg.model_cfg else asdict(ModelConfig()),
                    "epoch": epoch,
                    "val_auc": val_auc,
                },
                best_ckpt_path,
            )
            print(f"        ↑ Saved checkpoint (val_auc={val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {cfg.patience} epochs)")
                break

    # Save training history.
    history_path = Path(cfg.log_dir) / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)
    print(f"\nHistory saved to {history_path}")

    # --- Load best model and calibrate ---
    print(f"\nLoading best checkpoint (val_auc={best_val_auc:.4f})...")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    print("Fitting temperature scaling on validation set...")
    _, val_logits, val_labels = run_epoch(
        model, val_loader, criterion, None, device, is_train=False
    )
    scaler = TemperatureScaler(model)
    final_nll = scaler.fit(
        torch.from_numpy(val_logits).to(device),
        torch.from_numpy(val_labels).to(device),
    )
    print(f"  Temperature: {model.temperature.item():.4f}  (NLL after calib: {final_nll:.4f})")

    # Save the calibrated model.
    calibrated_path = Path(cfg.checkpoint_dir) / "calibrated_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "cfg": asdict(cfg.model_cfg) if cfg.model_cfg else asdict(ModelConfig()),
            "epoch": ckpt["epoch"],
            "val_auc": best_val_auc,
            "temperature": model.temperature.item(),
        },
        calibrated_path,
    )
    print(f"Calibrated model saved to {calibrated_path}")

    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ECG anomaly detector")
    parser.add_argument("--data_dir",        default="./ptbxl")
    parser.add_argument("--checkpoint_dir",  default="./checkpoints")
    parser.add_argument("--epochs",          type=int,   default=50)
    parser.add_argument("--batch_size",      type=int,   default=64)
    parser.add_argument("--lr",              type=float, default=3e-4)
    parser.add_argument("--num_workers",     type=int,   default=2)
    args = parser.parse_args()

    cfg = TrainConfig(**{k: v for k, v in vars(args).items()})
    train(cfg)
