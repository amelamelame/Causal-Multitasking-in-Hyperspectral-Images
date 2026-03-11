"""
evaluate.py  —  Standalone Evaluation Script
==============================================
Use this to:
  1. Evaluate any saved checkpoint on test set
  2. Generate all prediction maps without retraining

Run:
  python evaluate.py --checkpoint checkpoints/best.pth
  python evaluate.py --checkpoint checkpoints/best.pth --region full
"""

import os, sys, json, argparse
import numpy as np
import torch
import yaml
from torch.cuda.amp import autocast

from data.envi_reader   import TAIGAEnviReader, CAT_VARIABLES, REG_VARIABLES
from data.taiga_dataset import TAIGADataset, get_dataloaders, load_bad_stands
from models.baseline.model import MultitaskModel
from losses.loss         import MultitaskLoss
from evaluation.metrics  import MetricsTracker
from utils.gpu_utils     import get_device, VRAMMonitor
from visualization.prediction_maps import (
    predict_tile, get_gt_tile,
    plot_classification_maps, plot_regression_maps,
    plot_summary_dashboard, plot_training_curves
)


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    p   = cfg["paths"]

    device = get_device(cfg)
    vram   = VRAMMonitor(device)

    # ── Load data ──────────────────────────────────────────────
    print("\nLoading TAIGA files (memory-mapped)...")
    reader     = TAIGAEnviReader(p["hsi_file"], p["label_file"])
    bad_stands = load_bad_stands(p["bad_stands"])
    train_loader, val_loader, test_loader = get_dataloaders(reader, bad_stands, cfg)
    class_weights = train_loader.dataset.class_weights

    # ── Load model from checkpoint ─────────────────────────────
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ck     = torch.load(args.checkpoint, map_location=device)
    model  = MultitaskModel(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"  Trained for {ck.get('epoch','?')} epochs  |  best RMSE={ck.get('best',0):.4f}")
    vram.print("After model load")

    # ── Load history for curves ────────────────────────────────
    history = ck.get("history", None)

    # ── Test set evaluation ────────────────────────────────────
    print("\n" + "="*65)
    print("  TEST SET EVALUATION")
    print("="*65)
    loss_fn = MultitaskLoss(class_weights, cfg).to(device)
    tracker = MetricsTracker()
    running, n = 0.0, 0

    with torch.no_grad():
        from tqdm import tqdm
        for batch in tqdm(test_loader, desc="Evaluating test set", ncols=100):
            patches    = batch["patch"].to(device)
            cat_labels = {k: v.to(device) for k,v in batch["cat"].items()}
            reg_labels = {k: v.to(device) for k,v in batch["reg"].items()}
            with autocast(enabled=cfg["training"]["mixed_precision"]):
                out      = model(patches)
                loss, _  = loss_fn(out["cls"], out["reg"], cat_labels, reg_labels)
            tracker.update(out, batch)
            running += loss.item(); n += 1

    metrics = tracker.compute()
    print(tracker.summary(metrics))

    # Save metrics
    os.makedirs(p["results_dir"], exist_ok=True)
    out_json = os.path.join(p["results_dir"], "eval_metrics.json")
    with open(out_json, "w") as f:
        json.dump({k: round(float(v),6) for k,v in metrics.items()}, f, indent=2)
    print(f"\n  Metrics saved: {out_json}")

    # ── Prediction Maps ────────────────────────────────────────
    maps_dir = os.path.join(p["results_dir"], "maps")
    os.makedirs(maps_dir, exist_ok=True)
    vc       = cfg["viz"]

    # Choose tile region
    if args.region == "full":
        # Warning: full image is huge. Only do this on Colab with lots of RAM
        r0, r1 = 0, reader.lines
        c0, c1 = cfg["data"]["test_col_start"], reader.samples
        print("\n  ⚠  Full prediction — this may take 1+ hour on laptop!")
    else:
        r0, r1 = vc["map_row_start"], vc["map_row_end"]
        c0, c1 = vc["map_col_start"], vc["map_col_end"]

    print(f"\n  Generating maps for region rows={r0}:{r1}, cols={c0}:{c1} "
          f"({r1-r0}×{c1-c0} pixels)...")

    cls_pred, reg_pred, pred_mask = predict_tile(
        model, reader, r0, r1, c0, c1,
        patch_size = cfg["data"]["patch_size"],
        batch_size = 256,
        device     = device
    )
    cat_gt, reg_gt, gt_mask = get_gt_tile(reader, r0, r1, c0, c1)
    rgb  = reader.get_rgb(r0, r1, c0, c1)
    mask = pred_mask & gt_mask

    print("\n  Plotting classification maps...")
    plot_classification_maps(
        rgb, cat_gt, cls_pred, mask,
        os.path.join(maps_dir, "classification_maps.png"),
        title=f"OA={metrics['mean_OA']:.1f}%"
    )

    print("  Plotting regression maps...")
    plot_regression_maps(
        reg_gt, reg_pred, mask,
        os.path.join(maps_dir, "regression_maps.png"),
        title=f"Mean RMSE={metrics['mean_RMSE']:.4f}"
    )

    print("  Plotting summary dashboard...")
    plot_summary_dashboard(
        rgb, cat_gt, cls_pred, reg_gt, reg_pred, mask, metrics,
        os.path.join(maps_dir, "summary_dashboard.png"),
        title=f"OA={metrics['mean_OA']:.1f}%  RMSE={metrics['mean_RMSE']:.4f}"
    )

    if history:
        print("  Plotting training curves...")
        plot_training_curves(history, os.path.join(maps_dir, "training_curves.png"))

    print(f"\n✓ All maps saved to {maps_dir}/")
    print("  Files generated:")
    print("    classification_maps.png  ← GT vs Pred vs Error for 3 class variables")
    print("    regression_maps.png      ← GT vs Pred vs Error for 10 cont. variables")
    print("    summary_dashboard.png    ← One-page presentation slide")
    if history:
        print("    training_curves.png      ← Loss + OA + RMSE over epochs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pth")
    parser.add_argument("--region",     default="crop",
                        choices=["crop","full"],
                        help="'crop' = 1024x1024 tile (fast), 'full' = entire test area")
    main(parser.parse_args())
