"""Data loading and transforms - identical to Notebooks 1-3.

Every artifact loaded here was produced by Notebook 1 and MUST NOT be regenerated
(that would break reproducibility of the reported results).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from .config import (
    TARGET_SIZE, BATCH_SIZE, NUM_WORKERS, RANDOM_STATE, get_output_dir,
)
from .utils import seed_worker, make_generator


# ------------------------------------------------------------------ assets ----

def load_frozen_assets(output_dir: Path | None = None):
    """Load the frozen Notebook-1 artifacts. Returns a dict with keys:
    norm_mean, norm_std, class_names, num_classes, df_train, df_val, df_test.
    """
    out = Path(output_dir) if output_dir else get_output_dir()

    with open(out / "norm_stats.json") as f:
        _norm = json.load(f)
    norm_mean = tuple(_norm["mean"])
    norm_std = tuple(_norm["std"])

    with open(out / "label_mappings_binary.json") as f:
        _lm = json.load(f)
    class_names = [k for k, _ in sorted(_lm.items(), key=lambda kv: kv[1])]

    df_train = pd.read_csv(out / "train_metadata.csv")
    df_val = pd.read_csv(out / "val_metadata.csv")
    df_test = pd.read_csv(out / "test_metadata.csv")

    # Integrity: same contract Notebook 2 asserts on startup.
    for name, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
        if "image_path" not in d.columns or d["image_path"].isna().any():
            raise RuntimeError(f"{name}_metadata.csv missing populated image_path column")
        if "binary_encoded" not in d.columns:
            raise RuntimeError(f"{name}_metadata.csv missing binary_encoded column")
    tr = set(df_train["lesion_id"]); va = set(df_val["lesion_id"]); te = set(df_test["lesion_id"])
    if (tr & va) or (tr & te) or (va & te):
        raise RuntimeError("Lesion-level leakage detected across splits!")

    return {
        "norm_mean": norm_mean, "norm_std": norm_std,
        "class_names": class_names, "num_classes": len(class_names),
        "df_train": df_train, "df_val": df_val, "df_test": df_test,
    }


# -------------------------------------------------------------- transforms ----

def build_train_transform(norm_mean, norm_std) -> A.Compose:
    return A.Compose([
        A.Resize(*TARGET_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=20, p=0.6),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10),
            A.CLAHE(clip_limit=2.0),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 20.0)),
            A.GaussianBlur(blur_limit=3),
        ], p=0.2),
        A.CoarseDropout(max_holes=6, max_height=16, max_width=16, min_holes=1, p=0.3),
        A.Normalize(mean=norm_mean, std=norm_std),
        ToTensorV2(),
    ])


def build_eval_transform(norm_mean, norm_std) -> A.Compose:
    return A.Compose([
        A.Resize(*TARGET_SIZE),
        A.Normalize(mean=norm_mean, std=norm_std),
        ToTensorV2(),
    ])


def build_tta_transforms(norm_mean, norm_std) -> List[A.Compose]:
    """Deterministic four-view TTA: identity, H-flip, V-flip, 180-degree rotation."""
    tail = [A.Resize(*TARGET_SIZE), A.Normalize(mean=norm_mean, std=norm_std), ToTensorV2()]
    return [
        A.Compose(tail),
        A.Compose([A.HorizontalFlip(p=1.0)] + tail),
        A.Compose([A.VerticalFlip(p=1.0)] + tail),
        A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)] + tail),
    ]


# ---------------------------------------------------------------- dataset ----

class SkinLesionDataset(Dataset):
    """Reads images via the persisted `image_path` column from Notebook 1."""
    def __init__(self, dataframe: pd.DataFrame, transform: A.Compose, label_col: str = "binary_encoded"):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = np.array(Image.open(row["image_path"]).convert("RGB"))
        image = self.transform(image=image)["image"]
        return image, int(row[self.label_col])


# ------------------------------------------------------------ dataloaders ----

def _compute_class_sample_weights(df, label_col, power=0.5):
    counts = df[label_col].value_counts().to_dict()
    inv_freq = {c: (1.0 / n) ** power for c, n in counts.items()}
    return df[label_col].map(inv_freq).values


def build_dataloaders(assets: dict, label_col: str = "binary_encoded") -> Tuple[DataLoader, DataLoader, DataLoader]:
    tr, va, te = assets["df_train"], assets["df_val"], assets["df_test"]
    mean, std = assets["norm_mean"], assets["norm_std"]

    train_ds = SkinLesionDataset(tr, build_train_transform(mean, std), label_col)
    val_ds = SkinLesionDataset(va, build_eval_transform(mean, std), label_col)
    test_ds = SkinLesionDataset(te, build_eval_transform(mean, std), label_col)

    sample_weights = _compute_class_sample_weights(tr, label_col)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                    replacement=True, generator=make_generator(RANDOM_STATE))
    gen = make_generator(RANDOM_STATE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                              worker_init_fn=seed_worker, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, worker_init_fn=seed_worker)
    return train_loader, val_loader, test_loader
