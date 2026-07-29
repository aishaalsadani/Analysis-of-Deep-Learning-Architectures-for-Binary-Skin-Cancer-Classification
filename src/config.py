"""Frozen configuration - single source of truth.

These values MUST equal what was used in Notebooks 1-3. Any script that changes
them silently invalidates the leakage-safe split and the reported results.
"""
from pathlib import Path
import os

# -----------------------------------------------------------------------------
# Random state and split (FROZEN - inherited from Notebook 1)
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# -----------------------------------------------------------------------------
# Image pipeline (FROZEN)
# -----------------------------------------------------------------------------
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_WORKERS = 0

# -----------------------------------------------------------------------------
# Loss / regularization (FROZEN)
# -----------------------------------------------------------------------------
WARMUP_FRAC = 0.05
GRAD_CLIP_NORM = 1.0
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
MIX_PROB = 0.25
EMA_DECAY = 0.999
LABEL_SMOOTHING = 0.05
FOCAL_GAMMA = 2.0

# -----------------------------------------------------------------------------
# Default training hyperparameters (the experiment BASELINE point)
# -----------------------------------------------------------------------------
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 30           # matches EPOCHS_MAIN in NB2
PATIENCE = 8

# -----------------------------------------------------------------------------
# Binary mapping (FROZEN)
# -----------------------------------------------------------------------------
MALIGNANT_CLASSES = {"mel", "bcc", "akiec"}
BENIGN_CLASSES = {"nv", "bkl", "df", "vasc"}

# -----------------------------------------------------------------------------
# Paths - resolved from the SKIN_DATA_DIR env var, or fall back to the
# absolute path used during the executed run (documented in README.md).
# Users only need to set SKIN_DATA_DIR to their machine's data root.
# -----------------------------------------------------------------------------
_DEFAULT_DATA_DIR = r"D:\archive"

def get_data_dir() -> Path:
    return Path(os.environ.get("SKIN_DATA_DIR", _DEFAULT_DATA_DIR))

def get_output_dir() -> Path:
    """Where all processed artifacts live - CSVs, checkpoints, JSONs, figures."""
    out = Path(os.environ.get("SKIN_OUTPUT_DIR", str(get_data_dir() / "outputs")))
    out.mkdir(parents=True, exist_ok=True)
    return out
