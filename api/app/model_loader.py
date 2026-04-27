"""
Model loader — loads the trained ECGAnomalyDetector checkpoint once at
startup and holds a reference for the request handlers.

Design notes
------------
- Loaded once via FastAPI's lifespan context (api/app/main.py) so we don't
  pay the deserialize cost per request.
- Runs on CPU by default (Cloud Run containers typically don't have GPU).
  Override via ECG_DEVICE=cuda for GPU deployment.
- Thread-safe at inference time because PyTorch models are stateless
  once in eval() mode.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ecg_model.model import (
    CONDITION_NAMES,
    ECGAnomalyDetector,
    ModelConfig,
    load_checkpoint,
)


class ModelBundle:
    """Holds the loaded model plus metadata for request handlers."""

    def __init__(self, model: ECGAnomalyDetector, device: torch.device, path: str):
        self.model = model
        self.device = device
        self.checkpoint_path = path

    @torch.no_grad()
    def predict(self, signal: np.ndarray) -> np.ndarray:
        """Run a single inference.

        Parameters
        ----------
        signal : np.ndarray, shape (12, 5000)
            Preprocessed, z-scored, aligned signal from ECGPreprocessor.

        Returns
        -------
        probs : np.ndarray, shape (10,)
            Calibrated probabilities in [0, 1] for each condition.
        """
        if signal.shape != (12, 5000):
            raise ValueError(f"Expected shape (12, 5000), got {signal.shape}")

        x = torch.from_numpy(signal.astype(np.float32)).unsqueeze(0).to(self.device)
        probs = self.model.predict_proba(x)
        return probs[0].cpu().numpy()


_bundle: Optional[ModelBundle] = None


def get_model_bundle() -> ModelBundle:
    """Access the loaded model from request handlers."""
    if _bundle is None:
        raise RuntimeError(
            "Model not loaded. This should be called only after app startup."
        )
    return _bundle


def load_model(
    checkpoint_path: Optional[str] = None,
    device_str: Optional[str] = None,
) -> ModelBundle:
    """Load the checkpoint and set the module-level singleton.

    Called from main.py's lifespan startup.
    """
    global _bundle

    checkpoint_path = checkpoint_path or os.environ.get(
        "ECG_CHECKPOINT_PATH", "./model/calibrated_model.pt"
    )
    device_str = device_str or os.environ.get(
        "ECG_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
    )

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {path}. "
            f"Set ECG_CHECKPOINT_PATH env var or place calibrated_model.pt in api/model/"
        )

    device = torch.device(device_str)
    model = load_checkpoint(str(path), device=str(device))
    model.eval()

    _bundle = ModelBundle(model=model, device=device, path=str(path))
    return _bundle
