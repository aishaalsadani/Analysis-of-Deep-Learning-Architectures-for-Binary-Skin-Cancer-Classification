"""Shared training and evaluation engine for the Applied Deep Learning project.

The public API is the same as the code inside the notebooks:
    from src.engine import run_experiment, evaluate_checkpoint
    from src.data import load_split_assets, build_dataloaders
    from src.models import build_transfer_model, build_baseline_loaded

Nothing here re-implements notebook logic; it *is* the notebook logic, so a
notebook run and a CLI run produce byte-identical outputs.
"""
