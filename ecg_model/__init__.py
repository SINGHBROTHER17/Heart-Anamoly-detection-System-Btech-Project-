"""ecg_model — Phase 3 ML training package."""
from .model import ECGAnomalyDetector, ModelConfig, CONDITION_NAMES, N_CONDITIONS, build_model, load_checkpoint
from .dataset import PTBXLDataset, load_ptbxl, make_dataloaders, download_ptbxl
from .train import TrainConfig, train
from .evaluate import evaluate, plot_training_history

__all__ = [
    "ECGAnomalyDetector", "ModelConfig", "CONDITION_NAMES", "N_CONDITIONS",
    "build_model", "load_checkpoint",
    "PTBXLDataset", "load_ptbxl", "make_dataloaders", "download_ptbxl",
    "TrainConfig", "train",
    "evaluate", "plot_training_history",
]
