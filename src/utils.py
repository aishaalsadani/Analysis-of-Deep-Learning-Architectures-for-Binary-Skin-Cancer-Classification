"""Reproducibility utilities - identical to Notebooks 1-3.

DO NOT change the internals. These are the project-wide standard: any run that
uses different seeding is not comparable with the reported results.
"""
import os
import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed every RNG we rely on for deterministic behaviour."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Per-DataLoader-worker seeding (required for reproducible augmentation)."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g
