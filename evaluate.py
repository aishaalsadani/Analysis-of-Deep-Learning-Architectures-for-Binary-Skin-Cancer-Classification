#!/usr/bin/env python3
"""evaluate.py - score a trained checkpoint against the frozen Notebook-1 test split.

Loads the frozen assets (split CSVs, normalisation, label mapping), rebuilds the
requested model, applies the SAME evaluation protocol as Notebook 3 (deterministic
4-view TTA + recall-constrained threshold picked on validation only), and writes:

    outputs/predictions_<exp_name>.csv
    outputs/eval_<exp_name>.json

Examples
--------
    # Score any of the checkpoints Notebook 2 produced
    python evaluate.py --model resnet50 --checkpoint outputs/best_model_ResNet50.pt
    python evaluate.py --model densenet121 --checkpoint outputs/best_model_DenseNet121.pt

    # Score Notebook 1's from-scratch baseline
    python evaluate.py --model baseline

    # Score the "final" model committed as the project's headline result
    # (ResNet50, AdamW, lr=1e-4; see README)
    python evaluate.py --model resnet50 --checkpoint outputs/best_model_ResNet50_final.pt

The reported numbers exactly match Notebook 3's ranking table when the same
checkpoint is used, because the evaluation code is shared.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.config import BATCH_SIZE, NUM_WORKERS, get_output_dir
from src.utils import seed_worker
from src.data import load_frozen_assets, build_eval_transform, SkinLesionDataset
from src.models import (
    build_transfer_from_checkpoint, build_baseline_loaded, count_params, TRANSFER_BACKBONES,
)
from src.engine import (
    DEVICE, tta_predict_probs, choose_threshold, metrics_from_probs,
    measure_inference_time, save_predictions_csv, save_result_json,
)


_BACKBONE_ALIASES = {
    "baseline": "baseline",
    "cnn": "baseline",
    "cnn-baseline": "baseline",
    "efficientnet": "efficientnet_b0",
    "efficientnet-b0": "efficientnet_b0",
    "efficientnetb0": "efficientnet_b0",
    "efficientnet_b0": "efficientnet_b0",
    "resnet": "resnet50", "resnet50": "resnet50", "resnet-50": "resnet50",
    "densenet": "densenet121", "densenet121": "densenet121", "densenet-121": "densenet121",
}


def _resolve(name: str) -> str:
    key = name.lower().strip()
    if key not in _BACKBONE_ALIASES:
        raise SystemExit(
            f"Unknown --model {name!r}. Choose one of: baseline, "
            f"{sorted(set(TRANSFER_BACKBONES))}"
        )
    return _BACKBONE_ALIASES[key]


def _safe(name: str) -> str:
    return name.replace(" ", "").replace("/", "-")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint on the frozen test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True,
                   help="baseline | efficientnet_b0 | resnet50 | densenet121")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Path to a .pt with 'ema' or 'model' state dict. "
                        "For --model baseline, defaults to outputs/best_model_binary.pt.")
    p.add_argument("--exp-name", default=None,
                   help="Experiment name (used for output filenames). "
                        "Defaults to the resolved model name.")
    p.add_argument("--min-recall", type=float, default=0.88,
                   help="Malignant recall floor for the validation-picked threshold.")
    p.add_argument("--threshold", type=float, default=None,
                   help="Skip validation threshold search and use this fixed value.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Override the frozen output dir.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    resolved = _resolve(args.model)
    exp_name = args.exp_name or resolved
    out_dir = args.output_dir if args.output_dir else get_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    assets = load_frozen_assets(out_dir)
    print(f"evaluate.py | model={resolved} device={DEVICE}")
    print(f"             test samples={len(assets['df_test'])} output_dir={out_dir}")

    # ---- rebuild + load weights
    if resolved == "baseline":
        ckpt = args.checkpoint or (out_dir / "best_model_binary.pt")
        model = build_baseline_loaded(num_classes=assets["num_classes"], checkpoint=ckpt).to(DEVICE)
    else:
        if not args.checkpoint:
            # Default to NB2's naming convention.
            default_names = {"efficientnet_b0": "EfficientNet-B0",
                             "resnet50": "ResNet50", "densenet121": "DenseNet121"}
            ckpt = out_dir / f"best_model_{default_names[resolved]}.pt"
        else:
            ckpt = args.checkpoint
        if not Path(ckpt).exists():
            raise SystemExit(f"Checkpoint not found: {ckpt}")
        model = build_transfer_from_checkpoint(resolved, ckpt,
                                                num_classes=assets["num_classes"]).to(DEVICE)
    n_params = count_params(model)

    # ---- threshold (val only) then test evaluation with TTA
    threshold = args.threshold if args.threshold is not None \
                else choose_threshold(model, assets, min_recall=args.min_recall)
    probs = tta_predict_probs(model, assets["df_test"], assets)
    labels = assets["df_test"]["binary_encoded"].values
    m, preds = metrics_from_probs(labels, probs, threshold)

    # Inference timing on a plain (no-TTA) test loader.
    test_ds = SkinLesionDataset(assets["df_test"],
                                 build_eval_transform(assets["norm_mean"], assets["norm_std"]))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, worker_init_fn=seed_worker)
    infer_ms = measure_inference_time(model, test_loader)

    # ---- save
    save_predictions_csv(out_dir / f"predictions_{_safe(exp_name)}.csv",
                          assets["df_test"]["image_id"].values, labels, preds, probs)
    record = {"model": exp_name, "backbone": resolved,
              "checkpoint": str(ckpt) if resolved != "baseline" or args.checkpoint
                            else str(out_dir / "best_model_binary.pt"),
              "threshold": float(threshold), **m,
              "infer_ms_per_img": infer_ms, "n_params": n_params}
    save_result_json(out_dir / f"eval_{_safe(exp_name)}.json", record)

    print(f"[eval] {exp_name} | acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
          f"AUC={m['roc_auc']:.4f} recall={m['recall']:.4f} precision={m['precision']:.4f} "
          f"| threshold={threshold:.2f} params={n_params/1e6:.2f}M "
          f"infer={infer_ms:.2f}ms/img")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
