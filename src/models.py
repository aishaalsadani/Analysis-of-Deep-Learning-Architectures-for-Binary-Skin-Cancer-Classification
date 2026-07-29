"""Model definitions - identical to Notebooks 1-2.

- `SkinLesionResNetSE`: the from-scratch baseline (used in NB1).
- `build_transfer_model`: EfficientNet-B0 / ResNet50 / DenseNet121 with a two-logit
  head (used in NB2 §3).
- `build_baseline_loaded`: reconstructs the baseline and loads NB1's EMA weights.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models

from .config import get_output_dir


# ============================================================================
# From-scratch baseline (identical to Notebook 1)
# ============================================================================

def _drop_path(x, drop_prob, training):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()
    return x.div(keep_prob) * mask


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__(); self.drop_prob = drop_prob
    def forward(self, x):
        return _drop_path(x, self.drop_prob, self.training)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8)), nn.ReLU(inplace=False),
            nn.Linear(max(channels // reduction, 8), channels), nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        return x * self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)


class ResidualSEBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, drop_path_prob=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)
        self.drop_path = DropPath(drop_path_prob)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))
    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=False)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.drop_path(out)
        return F.relu(out + identity, inplace=False)


class SkinLesionResNetSE(nn.Module):
    """Custom ResNet-SE trained fully from scratch (baseline)."""
    def __init__(self, num_classes: int, base_width=24, dropout=0.3, max_drop_path=0.2):
        super().__init__()
        widths = [base_width, base_width*2, base_width*4, base_width*8, base_width*16]
        blocks_per_stage = [2, 2, 3, 3]
        depth_probs = np.linspace(0, max_drop_path, sum(blocks_per_stage)).tolist()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=False))
        stages, in_ch, idx = [], widths[0], 0
        for stage_i, n_blocks in enumerate(blocks_per_stage):
            out_ch = widths[stage_i + 1]
            for b in range(n_blocks):
                stride = 2 if b == 0 else 1
                stages.append(ResidualSEBlock(in_ch, out_ch, stride=stride,
                                              drop_path_prob=depth_probs[idx]))
                in_ch = out_ch; idx += 1
        self.stages = nn.Sequential(*stages)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                         nn.Linear(in_ch, num_classes))
    def forward(self, x):
        return self.classifier(self.global_pool(self.stages(self.stem(x))))


def init_weights(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02); nn.init.zeros_(m.bias)


def build_baseline_loaded(num_classes: int = 2, checkpoint: Path | None = None) -> nn.Module:
    """Reconstruct the baseline and load Notebook 1's EMA weights."""
    ckpt_path = Path(checkpoint) if checkpoint else (get_output_dir() / "best_model_binary.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found: {ckpt_path}. Run Notebook 1 first."
        )
    model = SkinLesionResNetSE(num_classes=num_classes)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["ema"])
    return model


# ============================================================================
# Transfer-learning backbones (identical to Notebook 2 §3)
# ============================================================================

TRANSFER_BACKBONES = ("efficientnet_b0", "resnet50", "densenet121")


def build_transfer_model(backbone: str, num_classes: int = 2, head_dropout: float = 0.3) -> nn.Module:
    """Build a torchvision backbone with a two-logit classification head.
    Weights are ImageNet-pretrained (matches NB2's TRANSFER_BUILDERS)."""
    backbone = backbone.lower()
    if backbone == "efficientnet_b0":
        m = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(in_f, num_classes))
    elif backbone == "resnet50":
        m = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(in_f, num_classes))
    elif backbone == "densenet121":
        m = tv_models.densenet121(weights=tv_models.DenseNet121_Weights.IMAGENET1K_V1)
        in_f = m.classifier.in_features
        m.classifier = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(in_f, num_classes))
    else:
        raise ValueError(f"Unknown backbone: {backbone!r} (expected one of {TRANSFER_BACKBONES})")
    return m


def build_transfer_from_checkpoint(backbone: str, checkpoint: Path,
                                    num_classes: int = 2) -> nn.Module:
    """Rebuild a transfer model and load its EMA weights from a NB2 checkpoint."""
    model = build_transfer_model(backbone, num_classes=num_classes)
    ckpt = torch.load(Path(checkpoint), map_location="cpu")
    key = "ema" if "ema" in ckpt else ("model" if "model" in ckpt else None)
    if key is None:
        raise RuntimeError(f"Checkpoint {checkpoint} has neither 'ema' nor 'model' key")
    model.load_state_dict(ckpt[key])
    return model


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
