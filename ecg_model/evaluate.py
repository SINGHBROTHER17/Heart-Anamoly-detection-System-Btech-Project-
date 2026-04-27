"""
evaluate.py — Model Evaluation
================================

Loads a saved checkpoint, runs it on the held-out test set, and produces:
  - Per-class AUC-ROC and AUPRC
  - Sensitivity (recall) and specificity at threshold 0.5
  - Calibration curves (reliability diagrams) before and after temperature scaling
  - Expected Calibration Error (ECE)
  - Training history curves (loss, AUC over epochs)
  - Confusion matrix heatmap for each condition

All plots are saved to --output_dir as PNGs, and a JSON summary is written
for machine consumption by the FastAPI layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from .model import (
    CONDITION_NAMES,
    ECGAnomalyDetector,
    N_CONDITIONS,
    ModelConfig,
    load_checkpoint,
)
from .dataset import load_ptbxl, make_dataloaders
from .train import TrainConfig, run_epoch, compute_metrics, build_pos_weight


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------

def expected_calibration_error(
    probs: np.ndarray,    # (N,) predicted probabilities
    labels: np.ndarray,   # (N,) binary ground truth
    n_bins: int = 10,
) -> float:
    """Compute ECE via equal-width probability bins."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def compute_ece_per_class(
    probs: np.ndarray,    # (N, 10)
    labels: np.ndarray,   # (N, 10)
    n_bins: int = 10,
) -> dict[str, float]:
    return {
        cond: expected_calibration_error(probs[:, i], labels[:, i], n_bins)
        for i, cond in enumerate(CONDITION_NAMES)
    }


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint_path: str,
    data_dir: str = "./ptbxl",
    output_dir: str = "./eval_output",
    batch_size: int = 64,
    num_workers: int = 2,
    device_str: str | None = None,
) -> dict:
    """Run full evaluation on the test set. Returns summary dict."""
    device = torch.device(
        device_str or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    model = load_checkpoint(checkpoint_path, device=str(device))
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Temperature: {model.temperature.item():.4f}")

    # --- Load data ---
    _, _, test_ds = load_ptbxl(data_dir)
    *_, test_loader = make_dataloaders(
        *load_ptbxl(data_dir),
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # --- Collect raw logits (uncalibrated) ---
    criterion = nn.BCEWithLogitsLoss()
    test_loss, raw_logits, all_labels = run_epoch(
        model, test_loader, criterion, None, device, is_train=False
    )
    raw_probs = 1.0 / (1.0 + np.exp(-raw_logits))          # sigmoid

    # --- Calibrated probs (temperature already baked into model.temperature) ---
    T = model.temperature.item()
    cal_probs = 1.0 / (1.0 + np.exp(-raw_logits / T))

    # --- Metrics ---
    preds = (cal_probs >= 0.5).astype(int)
    summary: dict = {
        "test_loss": float(test_loss),
        "per_condition": {},
    }

    for i, cond in enumerate(CONDITION_NAMES):
        pos = int(all_labels[:, i].sum())
        neg = len(all_labels) - pos
        if pos == 0:
            summary["per_condition"][cond] = {
                "note": "no positive examples in test set",
                "n_pos": 0, "n_neg": neg,
            }
            continue

        auc_roc = roc_auc_score(all_labels[:, i], cal_probs[:, i])
        auprc   = average_precision_score(all_labels[:, i], cal_probs[:, i])

        tp = int(((preds[:, i] == 1) & (all_labels[:, i] == 1)).sum())
        tn = int(((preds[:, i] == 0) & (all_labels[:, i] == 0)).sum())
        fp = int(((preds[:, i] == 1) & (all_labels[:, i] == 0)).sum())
        fn = int(((preds[:, i] == 0) & (all_labels[:, i] == 1)).sum())

        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)

        ece_before = expected_calibration_error(raw_probs[:, i], all_labels[:, i])
        ece_after  = expected_calibration_error(cal_probs[:, i], all_labels[:, i])

        summary["per_condition"][cond] = {
            "auc_roc":     round(auc_roc, 4),
            "auprc":       round(auprc, 4),
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "ece_before":  round(ece_before, 4),
            "ece_after":   round(ece_after, 4),
            "n_pos": pos,
            "n_neg": neg,
        }

    # Macro averages.
    valid = [v for v in summary["per_condition"].values() if "auc_roc" in v]
    summary["macro_auc_roc"]  = round(np.mean([v["auc_roc"]  for v in valid]), 4)
    summary["macro_auprc"]    = round(np.mean([v["auprc"]    for v in valid]), 4)
    summary["macro_ece_before"] = round(np.mean([v["ece_before"] for v in valid]), 4)
    summary["macro_ece_after"]  = round(np.mean([v["ece_after"]  for v in valid]), 4)

    print("\n=== Test Set Results ===")
    print(f"  Macro AUC-ROC : {summary['macro_auc_roc']:.4f}")
    print(f"  Macro AUPRC   : {summary['macro_auprc']:.4f}")
    print(f"  ECE (before)  : {summary['macro_ece_before']:.4f}")
    print(f"  ECE (after)   : {summary['macro_ece_after']:.4f}")
    print(f"\n  {'Condition':<42} AUC-ROC  AUPRC  Sens   Spec  ECE→")
    for cond, m in summary["per_condition"].items():
        if "auc_roc" not in m:
            print(f"  {cond:<42} (no positives)")
            continue
        print(
            f"  {cond:<42} {m['auc_roc']:.3f}    {m['auprc']:.3f}  "
            f"{m['sensitivity']:.3f}  {m['specificity']:.3f}  {m['ece_after']:.3f}"
        )

    # Save summary JSON.
    summary_path = Path(output_dir) / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # --- Plots ---
    if HAS_MPL:
        _plot_calibration(raw_probs, cal_probs, all_labels, output_dir)
        _plot_roc_curves(cal_probs, all_labels, output_dir)
        print(f"Plots saved to {output_dir}/")

    return summary


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_calibration(
    raw_probs: np.ndarray,
    cal_probs: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    n_bins: int = 10,
):
    """Reliability diagrams: one subplot per condition, before vs after calibration."""
    n_conds = len(CONDITION_NAMES)
    ncols = 5
    nrows = (n_conds + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = np.array(axes).flatten()

    for i, cond in enumerate(CONDITION_NAMES):
        ax = axes[i]
        if labels[:, i].sum() == 0:
            ax.set_title(f"{cond}\n(no positives)")
            ax.axis("off")
            continue

        frac_pos_raw, mean_pred_raw = calibration_curve(
            labels[:, i], raw_probs[:, i], n_bins=n_bins, strategy="uniform"
        )
        frac_pos_cal, mean_pred_cal = calibration_curve(
            labels[:, i], cal_probs[:, i], n_bins=n_bins, strategy="uniform"
        )

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect")
        ax.plot(mean_pred_raw, frac_pos_raw, "b-o", markersize=4, label="Before calib")
        ax.plot(mean_pred_cal, frac_pos_cal, "r-o", markersize=4, label="After calib")

        ece_before = expected_calibration_error(raw_probs[:, i], labels[:, i])
        ece_after  = expected_calibration_error(cal_probs[:, i], labels[:, i])

        ax.set_title(f"{cond}\nECE {ece_before:.3f}→{ece_after:.3f}", fontsize=8)
        ax.set_xlabel("Mean predicted prob", fontsize=7)
        ax.set_ylabel("Fraction positive", fontsize=7)
        ax.legend(fontsize=6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Hide unused axes.
    for j in range(n_conds, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Reliability Diagrams (Calibration Curves)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "calibration_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_roc_curves(
    cal_probs: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
):
    """ROC curves for each condition on a single figure."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)

    colors = plt.cm.tab10(np.linspace(0, 1, N_CONDITIONS))
    for i, (cond, col) in enumerate(zip(CONDITION_NAMES, colors)):
        if labels[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(labels[:, i], cal_probs[:, i])
        auc = roc_auc_score(labels[:, i], cal_probs[:, i])
        short = cond.replace("Left ", "L-").replace("Right ", "R-")
        ax.plot(fpr, tpr, color=col, linewidth=1.5, label=f"{short} (AUC={auc:.2f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Calibrated Model")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "roc_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_training_history(history_path: str, output_dir: str):
    """Load history.json and plot loss + AUC curves."""
    if not HAS_MPL:
        return

    with open(history_path) as f:
        history = json.load(f)

    epochs     = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"] for h in history]
    val_auc    = [h["val_auc"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_loss, label="Train Loss")
    ax1.plot(epochs, val_loss,   label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()

    ax2.plot(epochs, val_auc, color="green", label="Val Macro AUC-ROC")
    best_epoch = epochs[int(np.argmax(val_auc))]
    best_auc   = max(val_auc)
    ax2.axvline(best_epoch, color="gray", linestyle="--", linewidth=0.8,
                label=f"Best epoch {best_epoch} (AUC={best_auc:.3f})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro AUC-ROC")
    ax2.set_title("Validation AUC-ROC")
    ax2.legend()
    ax2.set_ylim(0.5, 1.0)

    fig.tight_layout()
    fig.savefig(Path(output_dir) / "training_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved to {output_dir}/training_curves.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ECG anomaly detector")
    parser.add_argument("--checkpoint", required=True, help="Path to calibrated_model.pt")
    parser.add_argument("--data_dir",   default="./ptbxl")
    parser.add_argument("--output_dir", default="./eval_output")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
