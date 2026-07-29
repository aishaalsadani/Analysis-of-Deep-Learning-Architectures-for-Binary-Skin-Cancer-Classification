"""Training and evaluation engine.

Everything here is identical to Notebooks 1 and 2 - MixUp/CutMix, Focal loss, EMA,
warmup-cosine schedule, AMP training loop, deterministic 4-view TTA, and the
recall-constrained threshold selection. Kept in one place so the CLI scripts
and the notebooks share the same code path (no duplicated logic).
"""
from __future__ import annotations
import copy, math, time, json as _json
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

from .config import (
    BATCH_SIZE, NUM_WORKERS, WARMUP_FRAC, GRAD_CLIP_NORM,
    MIXUP_ALPHA, CUTMIX_ALPHA, MIX_PROB, EMA_DECAY,
)
from .utils import seed_worker
from .data import build_tta_transforms, SkinLesionDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- augment ----

def _rand_bbox(size, lam):
    H, W = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_rat), int(W * cut_rat)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = np.clip(cy - cut_h // 2, 0, H), np.clip(cy + cut_h // 2, 0, H)
    x1, x2 = np.clip(cx - cut_w // 2, 0, W), np.clip(cx + cut_w // 2, 0, W)
    return x1, y1, x2, y2


def _mixup_cutmix(images, labels):
    if np.random.rand() > MIX_PROB:
        return images, labels, labels, 1.0
    perm = torch.randperm(images.size(0), device=images.device)
    labels_b = labels[perm]
    if np.random.rand() < 0.5:
        lam = np.random.beta(CUTMIX_ALPHA, CUTMIX_ALPHA)
        x1, y1, x2, y2 = _rand_bbox(images.size(), lam)
        images[:, :, y1:y2, x1:x2] = images[perm][:, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
    else:
        lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
        images = lam * images + (1 - lam) * images[perm]
    return images, labels, labels_b, float(lam)


# ------------------------------------------------------------------- loss ----

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha, self.gamma, self.label_smoothing = alpha, gamma, label_smoothing
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()
    def forward_soft(self, logits, targets_a, targets_b, lam):
        return lam * self.forward(logits, targets_a) + (1 - lam) * self.forward(logits, targets_b)


# -------------------------------------------------------------------- EMA ----

class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.params = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.params[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for n, b in model.named_buffers():
            self.buffers[n] = b.detach().clone()
    def state_dict(self):
        return {**self.params, **self.buffers}
    def apply_to(self, model):
        model.load_state_dict(self.state_dict(), strict=True)


# ---------------------------------------------------------------- optim ------

def build_optimizer(name, params, lr, weight_decay):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    raise ValueError(f"Unknown optimizer: {name!r}")


def build_scheduler(optimizer, steps_per_epoch, num_epochs, warmup_frac=WARMUP_FRAC):
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = max(1, int(total_steps * warmup_frac))
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return LambdaLR(optimizer, lr_lambda)


# -------------------------------------------------------------- train loop ---

def _train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, ema):
    model.train()
    running_loss, clean_correct, clean_total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        images, targets_a, targets_b, lam = _mixup_cutmix(images, labels)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
            outputs = model(images)
            loss = criterion.forward_soft(outputs, targets_a, targets_b, lam)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)
        running_loss += loss.item() * images.size(0)
        if lam == 1.0:
            clean_correct += (outputs.argmax(1) == labels).sum().item()
            clean_total += labels.size(0)
    return (running_loss / len(loader.dataset),
            clean_correct / clean_total if clean_total else float("nan"))


@torch.no_grad()
def _validate(model, loader, criterion):
    model.eval()
    running_loss, all_preds, all_labels = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        with autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        all_preds.append(outputs.argmax(1).cpu())
        all_labels.append(labels.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return (running_loss / len(loader.dataset),
            accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds, average="macro"))


def fit(model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs, patience, checkpoint_path, verbose=True):
    """Train with early stopping on val macro-F1. Returns (best_val_f1, history_df)."""
    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))
    ema = EMA(model, EMA_DECAY)
    ema_model = copy.deepcopy(model)
    best_val_f1, no_improve, history = 0.0, 0, []
    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = _train_one_epoch(model, train_loader, criterion,
                                            optimizer, scheduler, scaler, ema)
        ema.apply_to(ema_model)
        vl_loss, vl_acc, vl_f1 = _validate(ema_model, val_loader, criterion)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": vl_loss, "val_acc": vl_acc, "val_macro_f1": vl_f1})
        if verbose:
            print(f"  epoch {epoch:3d}/{num_epochs} | train_loss={tr_loss:.4f} "
                  f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f} val_f1={vl_f1:.4f}")
        if vl_f1 > best_val_f1:
            best_val_f1, no_improve = vl_f1, 0
            torch.save({"model": model.state_dict(), "ema": ema.state_dict()}, checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  early stopping at epoch {epoch}")
                break
    return best_val_f1, pd.DataFrame(history)


# --------------------------------------------------- inference / metrics -----

@torch.no_grad()
def tta_predict_probs(model, dataframe, assets, label_col="binary_encoded"):
    """4-view TTA softmax probabilities. Deterministic; matches NB2 exactly."""
    model.eval()
    probs = np.zeros((len(dataframe), assets["num_classes"]), dtype=np.float32)
    for t in build_tta_transforms(assets["norm_mean"], assets["norm_std"]):
        ds = SkinLesionDataset(dataframe, t, label_col)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, worker_init_fn=seed_worker)
        chunks = []
        for images, _ in loader:
            images = images.to(DEVICE)
            with autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                logits = model(images)
            chunks.append(F.softmax(logits, dim=1).float().cpu().numpy())
        probs += np.concatenate(chunks, axis=0)
    return probs / 4.0


def choose_threshold(model, assets, min_recall: float = 0.88) -> float:
    """Recall-constrained best-macro-F1 threshold chosen on VALIDATION only.
    Matches NB1's protocol byte-for-byte."""
    df_val = assets["df_val"]
    val_probs = tta_predict_probs(model, df_val, assets)
    y_val = df_val["binary_encoded"].values
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        preds_t = (val_probs[:, 1] >= t).astype(int)
        if recall_score(y_val, preds_t, pos_label=1) < min_recall:
            continue
        f1 = f1_score(y_val, preds_t, average="macro")
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def metrics_from_probs(labels, probs, threshold):
    preds = (probs[:, 1] >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    m = {
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall": recall_score(labels, preds, pos_label=1, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "f1": f1_score(labels, preds, pos_label=1, zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "roc_auc": roc_auc_score(labels, probs[:, 1]),
    }
    return m, preds


def measure_inference_time(model, loader) -> float:
    """Milliseconds per image over the loader (single view, no TTA)."""
    model.eval()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0, n = time.time(), 0
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(DEVICE)
            with autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                _ = model(images)
            n += images.size(0)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) * 1000.0 / max(n, 1)


def peak_gpu_mem_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1e6 if DEVICE.type == "cuda" else float("nan")


# ---------------------------------------------------- I/O for CLI scripts ----

def save_predictions_csv(path: Path, image_ids, labels, preds, probs) -> None:
    pd.DataFrame({
        "image_id": image_ids,
        "true_label": np.asarray(labels).astype(int),
        "predicted_label": np.asarray(preds).astype(int),
        "probability_benign": (1 - probs[:, 1]).astype(float),
        "probability_malignant": probs[:, 1].astype(float),
    }).sort_values("image_id").to_csv(path, index=False)


def save_result_json(path: Path, result: dict) -> None:
    with open(path, "w") as f:
        _json.dump(result, f, indent=2, default=str)
