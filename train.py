#!/usr/bin/env python3
"""train.py - reproduce any training run from the executed notebooks.

Loads the frozen Notebook-1 assets (split CSVs, normalisation, label mapping),
trains the requested model through the SAME engine the notebooks use, and writes
the same artifacts a Notebook-2 run would produce:

    outputs/best_model_<exp_name>.pt
    outputs/history_<exp_name>.csv
    outputs/predictions_<exp_name>.csv
    outputs/result_<exp_name>.json

Examples
--------
    # The best transfer-learning architecture from NB2 §4 (ResNet50, AdamW, 3e-4, 30 epochs)
    python train.py --model resnet50

    # The chosen final configuration (see README - ResNet50, AdamW, lr=1e-4, wd=0)
    python train.py --model resnet50 --lr 1e-4 --weight-decay 0.0 --exp-name ResNet50_final

    # An optimizer-study cell (ResNet50, AdamW, 10 epochs)
    python train.py --model resnet50 --optimizer adamw --epochs 10 --exp-name ResNet50_adamw

Notes
-----
The from-scratch baseline (Notebook 1's `SkinLesionResNetSE`) is intentionally
NOT retrainable through this script: retraining it here would take ~60 epochs
and reproduce Notebook 1's slow work-package.  If you need it, run Notebook 1;
`evaluate.py --model baseline` uses the checkpoint it produces.
"""
from __future__ import annotations
import argparse
import time
import sys
from pathlib import Path

import torch

# Make `src/` importable when running the script directly.
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.config import (
    LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE, RANDOM_STATE,
    FOCAL_GAMMA, LABEL_SMOOTHING, get_output_dir,
)
from src.utils import seed_everything
from src.data import load_frozen_assets, build_dataloaders
from src.models import build_transfer_model, count_params
from src.engine import (
    DEVICE, FocalLoss, build_optimizer, build_scheduler, fit,
    tta_predict_probs, choose_threshold, metrics_from_probs,
    measure_inference_time, peak_gpu_mem_mb,
    save_predictions_csv, save_result_json,
)

_BACKBONE_ALIASES = {
    "efficientnet": "efficientnet_b0",
    "efficientnet-b0": "efficientnet_b0",
    "efficientnetb0": "efficientnet_b0",
    "efficientnet_b0": "efficientnet_b0",
    "resnet": "resnet50", "resnet50": "resnet50", "resnet-50": "resnet50",
    "densenet": "densenet121", "densenet121": "densenet121", "densenet-121": "densenet121",
}


def _resolve_backbone(name: str) -> str:
    key = name.lower().strip()
    if key not in _BACKBONE_ALIASES:
        raise SystemExit(
            f"Unknown --model {name!r}. Choose one of: "
            f"{sorted(set(_BACKBONE_ALIASES.values()))} "
            f"(baseline is not retrainable via this script - run Notebook 1)."
        )
    return _BACKBONE_ALIASES[key]


def _safe_name(name: str) -> str:
    return name.replace(" ", "").replace("/", "-")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a transfer-learning model on the Notebook-1 split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True,
                   help="Backbone: efficientnet_b0 | resnet50 | densenet121")
    p.add_argument("--optimizer", default="adamw", choices=["adam", "adamw", "sgd"])
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--min-recall", type=float, default=0.88,
                   help="Malignant recall floor for the validation-picked threshold.")
    p.add_argument("--exp-name", default=None,
                   help="Experiment name (used for output filenames). "
                        "Defaults to the resolved backbone name.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Override the frozen output dir (defaults to $SKIN_OUTPUT_DIR "
                        "or $SKIN_DATA_DIR/outputs).")
    p.add_argument("--verbose", action="store_true", help="Print per-epoch metrics.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    backbone = _resolve_backbone(args.model)
    exp_name = args.exp_name or backbone
    out_dir = args.output_dir if args.output_dir else get_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"train.py | model={backbone} optimizer={args.optimizer} lr={args.lr:g} "
          f"wd={args.weight_decay:g} epochs={args.epochs} seed={args.seed}")
    print(f"          output_dir={out_dir}  device={DEVICE}")

    seed_everything(args.seed)
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # --- data
    assets = load_frozen_assets(out_dir)
    train_loader, val_loader, test_loader = build_dataloaders(assets)
    print(f"          train={len(assets['df_train'])} val={len(assets['df_val'])} "
          f"test={len(assets['df_test'])}")

    # --- model + train
    model = build_transfer_model(backbone, num_classes=assets["num_classes"]).to(DEVICE)
    n_params = count_params(model)

    criterion = FocalLoss(alpha=None, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)
    scheduler = build_scheduler(optimizer, len(train_loader), args.epochs)

    ckpt_path = out_dir / f"best_model_{_safe_name(exp_name)}.pt"
    t0 = time.time()
    best_val_f1, history_df = fit(model, train_loader, val_loader, criterion,
                                   optimizer, scheduler, args.epochs, args.patience,
                                   ckpt_path, verbose=args.verbose)
    train_time = time.time() - t0
    history_df.to_csv(out_dir / f"history_{_safe_name(exp_name)}.csv", index=False)

    # Deploy EMA weights for evaluation (same rule as NB2).
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE)["ema"])

    # --- evaluate on test set
    threshold = choose_threshold(model, assets, min_recall=args.min_recall)
    probs = tta_predict_probs(model, assets["df_test"], assets)
    labels = assets["df_test"]["binary_encoded"].values
    m, preds = metrics_from_probs(labels, probs, threshold)
    infer_ms = measure_inference_time(model, test_loader)
    mem_mb = peak_gpu_mem_mb()

    # --- save artifacts
    save_predictions_csv(out_dir / f"predictions_{_safe_name(exp_name)}.csv",
                          assets["df_test"]["image_id"].values, labels, preds, probs)
    result = {"model": exp_name, "backbone": backbone,
              "optimizer": args.optimizer, "lr": args.lr, "weight_decay": args.weight_decay,
              "epochs_cap": args.epochs, "seed": args.seed, "threshold": threshold, **m,
              "train_time_s": train_time, "infer_ms_per_img": infer_ms,
              "n_params": n_params, "peak_mem_mb": mem_mb,
              "best_val_macro_f1": best_val_f1}
    save_result_json(out_dir / f"result_{_safe_name(exp_name)}.json", result)

    print(f"[done] {exp_name} | macroF1={m['macro_f1']:.4f} AUC={m['roc_auc']:.4f} "
          f"recall={m['recall']:.4f} | threshold={threshold:.2f} | "
          f"params={n_params/1e6:.2f}M train={train_time:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
