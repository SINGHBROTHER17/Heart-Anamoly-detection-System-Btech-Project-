"""Constants used across the preprocessing pipeline."""

from __future__ import annotations

# Canonical 12-lead order used everywhere downstream (model input, reports).
LEAD_ORDER: tuple[str, ...] = (
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)

LEAD_INDEX: dict[str, int] = {name: i for i, name in enumerate(LEAD_ORDER)}

# Target sampling rate. PTB-XL ships at 500 Hz; the acquisition hardware
# (ADS1292) is configured to 500 Hz SPS, so this is the natural target.
TARGET_FS: int = 500

# 10-second analysis window at 500 Hz -> 5000 samples.
WINDOW_SECONDS: float = 10.0
WINDOW_SAMPLES: int = int(WINDOW_SECONDS * TARGET_FS)

# Minimum acceptable Signal Quality Index. Below this the pipeline refuses
# to emit a result. Chosen to match the product requirement (SQI < 0.6).
SQI_THRESHOLD: float = 0.60

# Powerline frequency. 50 Hz for India/EU, 60 Hz for US. Override at runtime
# via ECGPreprocessor(powerline_hz=60) if deploying stateside.
POWERLINE_HZ: float = 50.0

# Bandpass corners (Hz). 0.5 Hz kills baseline drift without crushing ST
# segment morphology, 40 Hz keeps QRS detail while suppressing EMG.
BANDPASS_LOW: float = 0.5
BANDPASS_HIGH: float = 40.0
BANDPASS_ORDER: int = 4
