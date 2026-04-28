"""
model.py — ECG Anomaly Detection Model
=======================================

Architecture: Hybrid 1D-CNN + Transformer

Flow
----
  Input (B, 12, 5000)
      │
      ▼
  LeadEncoder  ──── processes each of the 12 leads independently
  (Conv1D blocks)   4 blocks: Conv1D → BatchNorm → ReLU → MaxPool1d
  Output: (B, 12, C, T')   where C=128 channels, T'=5000/16=312 frames
      │
      ▼
  Reshape + positional encoding
  (B, 12*T', C)  → each (lead, time) position is a token
      │
      ▼
  TransformerEncoder
  2 layers, 4 heads, feedforward dim 512
  Captures inter-lead and temporal context jointly
      │
      ▼
  Global average pool over token dimension → (B, C)
      │
      ▼
  MultiLabelHead
  Linear → Sigmoid  — 10 independent binary outputs
      │
      ▼
  CalibrationWrapper (temperature scaling, applied post-hoc)

Design decisions
----------------
- Multi-label (Sigmoid, not Softmax): a patient can simultaneously
  have LBBB AND bradycardia AND LVH.
- Leads encoded independently first: the CNN sees per-lead morphology
  (P/Q/R/S/T shape) before the Transformer sees cross-lead relationships.
  This matches clinical reasoning.
- Positional encoding is learnable (not sinusoidal): the 12 leads have
  fixed clinical positions; a learnable table lets the model encode that
  prior directly.
- Temperature scaling only: it's the simplest post-hoc calibration that
  reliably reduces ECE below 0.05 on held-out data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Conditions (output order is canonical — do not reorder)
# ---------------------------------------------------------------------------

CONDITION_NAMES: tuple[str, ...] = (
    "Normal Sinus Rhythm",
    "Atrial Fibrillation",
    "ST Elevation",
    "Left Bundle Branch Block",
    "Right Bundle Branch Block",
    "Left Ventricular Hypertrophy",
    "Bradycardia",
    "Tachycardia",
    "First Degree AV Block",
    "Premature Ventricular Contraction",
)
N_CONDITIONS = len(CONDITION_NAMES)


# ---------------------------------------------------------------------------
# Config dataclass (so hyperparams are one place)
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    n_leads: int = 12
    n_samples: int = 5000
    n_conditions: int = N_CONDITIONS

    # CNN backbone
    cnn_channels: tuple[int, ...] = (32, 64, 128, 128)
    cnn_kernel_sizes: tuple[int, ...] = (7, 5, 5, 3)
    cnn_pool_sizes: tuple[int, ...] = (2, 2, 2, 2)   # 5000 → 312 after 4 pools
    cnn_dropout: float = 0.1

    # Transformer
    d_model: int = 128           # must equal cnn_channels[-1]
    n_heads: int = 4
    n_transformer_layers: int = 2
    dim_feedforward: int = 512
    transformer_dropout: float = 0.1

    # Classification head
    head_hidden: int = 256
    head_dropout: float = 0.3


# ---------------------------------------------------------------------------
# CNN Lead Encoder
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv1d → BatchNorm1d → ReLU → MaxPool1d (one stage)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, pool: int, dropout: float):
        super().__init__()
        padding = kernel // 2  # same-ish padding to preserve frame count before pool
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LeadEncoder(nn.Module):
    """Process each lead's 1-D time series independently through 4 CNN blocks.

    Applies the same convolutional weights to every lead (weight sharing),
    treating each lead as an independent channel-1 signal. This is equivalent
    to a grouped convolution with groups=n_leads, which saves parameters while
    forcing the network to learn lead-agnostic morphological features.

    Input:  (B, n_leads, n_samples)
    Output: (B, n_leads, C, T') where C = cnn_channels[-1], T' = n_samples / prod(pools)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        channels = [1] + list(cfg.cnn_channels)
        self.blocks = nn.ModuleList([
            ConvBlock(channels[i], channels[i + 1], cfg.cnn_kernel_sizes[i],
                      cfg.cnn_pool_sizes[i], cfg.cnn_dropout)
            for i in range(len(cfg.cnn_channels))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_leads, n_samples)
        B, L, N = x.shape
        # Merge batch and lead dims so each lead is a separate item in the batch.
        x = x.reshape(B * L, 1, N)     # (B*L, 1, N_samples)
        for block in self.blocks:
            x = block(x)                # (B*L, C, T')
        _, C, T = x.shape
        x = x.reshape(B, L, C, T)      # (B, n_leads, C, T')
        return x


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class LearnablePositionalEncoding(nn.Module):
    """Learnable position table for (lead_index, time_frame) tokens.

    We create separate tables for lead identity and temporal position and
    add them together, giving the Transformer two independent axes to
    attend along.
    """

    def __init__(self, n_leads: int, max_frames: int, d_model: int):
        super().__init__()
        # One embedding per lead (0–11) and one per time frame (0–T'-1).
        self.lead_emb = nn.Embedding(n_leads, d_model)
        self.time_emb = nn.Embedding(max_frames, d_model)

    def forward(self, x: torch.Tensor, n_leads: int, n_frames: int) -> torch.Tensor:
        # x: (B, n_leads * n_frames, d_model)  already reshaped
        device = x.device
        leads = torch.arange(n_leads, device=device).repeat_interleave(n_frames)
        times = torch.arange(n_frames, device=device).repeat(n_leads)
        return x + self.lead_emb(leads) + self.time_emb(times)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class ECGAnomalyDetector(nn.Module):
    """
    Hybrid 1D-CNN + Transformer for 12-lead ECG multi-label classification.

    Forward pass returns raw logits (pre-sigmoid) for use with
    BCEWithLogitsLoss during training. Call .predict_proba() for calibrated
    probabilities at inference time.
    """

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()
        self.cfg = cfg

        # Stage 1: CNN backbone (per-lead feature extraction)
        self.lead_encoder = LeadEncoder(cfg)

        # Compute the number of time frames after 4 pooling operations.
        # We compute it analytically so T' is available for the pos-enc table.
        t_prime = cfg.n_samples
        for p in cfg.cnn_pool_sizes:
            t_prime = t_prime // p
        self.t_prime = t_prime  # 5000 // 16 = 312

        # Stage 2: Projection to d_model (in case cnn_channels[-1] != d_model)
        cnn_out_ch = cfg.cnn_channels[-1]
        self.input_proj = (
            nn.Identity()
            if cnn_out_ch == cfg.d_model
            else nn.Linear(cnn_out_ch, cfg.d_model)
        )

        # Positional encoding
        self.pos_enc = LearnablePositionalEncoding(
            n_leads=cfg.n_leads,
            max_frames=self.t_prime + 32,   # small buffer for odd-length inputs
            d_model=cfg.d_model,
        )

        # Stage 3: Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.transformer_dropout,
            batch_first=True,   # input is (B, seq, d_model)
            norm_first=True,    # pre-norm; more stable training for medical signals
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_transformer_layers,
            enable_nested_tensor=False,
        )

        # Stage 4: Classification head
        self.classifier = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.head_hidden),
            nn.LayerNorm(cfg.head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden, cfg.n_conditions),
            # No sigmoid here — BCEWithLogitsLoss is numerically safer.
        )

        # Temperature scaling parameter (post-hoc calibration).
        # Initialized to 1.0 (identity); learned by CalibrationWrapper.
        self.register_buffer("temperature", torch.ones(1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 12, 5000) — normalized ECG, one row per lead

        Returns
        -------
        logits : (B, N_CONDITIONS) — raw logits, pass through sigmoid for probabilities
        """
        B, L, N = x.shape

        # --- CNN backbone ---
        feats = self.lead_encoder(x)            # (B, L, C, T')
        _, _, C, T = feats.shape

        # Rearrange to token sequence: (B, L*T', C)
        feats = feats.permute(0, 1, 3, 2)       # (B, L, T', C)
        feats = feats.reshape(B, L * T, C)       # (B, L*T', C)

        # Project to d_model if needed
        feats = self.input_proj(feats)           # (B, L*T', d_model)

        # Add positional encoding
        feats = self.pos_enc(feats, n_leads=L, n_frames=T)

        # --- Transformer ---
        feats = self.transformer(feats)          # (B, L*T', d_model)

        # Global average pooling over token dimension
        feats = feats.mean(dim=1)               # (B, d_model)

        # --- Classification head ---
        logits = self.classifier(feats)         # (B, N_CONDITIONS)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return calibrated probabilities in [0, 1].

        Temperature scaling: divide logits by T before sigmoid.
        T > 1 softens the probabilities (better calibrated for overconfident models).
        """
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits / self.temperature.clamp(min=0.05))

    def parameter_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ---------------------------------------------------------------------------
# Temperature Scaling calibration
# ---------------------------------------------------------------------------

class TemperatureScaler:
    """Post-hoc calibration via temperature scaling (Guo et al. 2017).

    Learns a single scalar T that minimises NLL on the validation set by
    dividing all logits by T before applying sigmoid. This is not a model
    weight — we optimize it separately after training.

    Usage
    -----
        scaler = TemperatureScaler(model)
        scaler.fit(val_logits, val_labels)   # val_logits: (N, 10) raw logits
        model.temperature = scaler.temperature
    """

    def __init__(self, model: ECGAnomalyDetector):
        self.model = model

    def fit(
        self,
        logits: torch.Tensor,   # (N, n_conditions) — raw logits on val set
        labels: torch.Tensor,   # (N, n_conditions) — ground truth binary labels
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """Optimise temperature. Returns final NLL."""
        temperature = nn.Parameter(
            self.model.temperature.clone().requires_grad_(True)
        )
        optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
        criterion = nn.BCEWithLogitsLoss()

        def eval_step():
            optimizer.zero_grad()
            loss = criterion(logits / temperature.clamp(min=0.05), labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)

        # Write the learned temperature back into the model buffer.
        self.model.temperature.copy_(temperature.detach().clamp(min=0.05))
        final_nll = criterion(
            logits / self.model.temperature, labels
        ).item()
        return final_nll


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_model(cfg: ModelConfig | None = None, device: str | None = None) -> ECGAnomalyDetector:
    """Construct an untrained model and move it to the target device."""
    model = ECGAnomalyDetector(cfg)
    if device is not None:
        model = model.to(device)
    return model


def load_checkpoint(
    path: str,
    device: str = "cpu",
    cfg: ModelConfig | None = None,
) -> ECGAnomalyDetector:
    """Load a saved model checkpoint.

    Checkpoints are dicts with keys: 'model_state', 'cfg', 'epoch', 'val_auc'.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Support both wrapped checkpoints {"model_state": ..., "cfg": ...}
    # and bare state dicts saved directly with torch.save(model.state_dict(), path).
    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        cfg = cfg or ModelConfig(**{k: v for k, v in ckpt.get("cfg", {}).items()
                                    if k in {f.name for f in fields(ModelConfig)}})
    else:
        state_dict = ckpt
    model = ECGAnomalyDetector(cfg or ModelConfig()).to(device)
    model.load_state_dict(state_dict)
    return model
