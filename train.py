"""
train.py  —  Main Training Script
===================================
Run: python train.py
Resume: python train.py --resume checkpoints/best.pth

Optimized for RTX 3060 6GB / 16 GB RAM on real TAIGA dataset.
"""

import os, sys, time, argparse, json
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

# ── Local imports ──────────────────────────────────────────────
from data.envi_reader   import TAIGAEnviReader, CAT_VARIABLES, REG_VARIABLES
from data.taiga_dataset import TAIGADataset, get_dataloaders, load_bad_stands
from models.baseline.model import MultitaskModel
from losses.loss         import MultitaskLoss
from evaluation.metrics  import MetricsTracker
from utils.gpu_utils     import set_seed, get_device, VRAMMonitor


# ──────────────────────────────────────────────────────────────
def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_ckpt(path, model, opt, scaler, epoch, best, history):
    torch.save({"epoch":epoch,"model":model.state_dict(),
                "opt":opt.state_dict(),"scaler":scaler.state_dict(),
                "best":best,"history":history}, path)


def load_ckpt(path, model, opt, scaler):
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model"])
    if opt and "opt" in ck:   opt.load_state_dict(ck["opt"])
    if scaler and "scaler" in ck: scaler.load_state_dict(ck["scaler"])
    return ck.get("epoch",0), ck.get("best",float("inf")), ck.get("history",{})


# ──────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, loss_fn, opt, scaler,
                    device, cfg, epoch, writer):
    model.train()
    running, n = 0.0, 0
    pbar = tqdm(loader, desc=f"Ep {epoch:03d} TRAIN", leave=False, ncols=110)
    for batch in pbar:
        patches    = batch["patch"].to(device, non_blocking=True)
        cat_labels = {k: v.to(device) for k,v in batch["cat"].items()}
        reg_labels = {k: v.to(device) for k,v in batch["reg"].items()}

        opt.zero_grad(set_to_none=True)
        with autocast(enabled=cfg["training"]["mixed_precision"]):
            out       = model(patches)
            loss, det = loss_fn(out["cls"], out["reg"], cat_labels, reg_labels)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
        scaler.step(opt); scaler.update()

        running += loss.item(); n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg = running / max(n, 1)
    if writer: writer.add_scalar("Loss/train", avg, epoch)
    return avg


@torch.no_grad()
def validate(model, loader, loss_fn, tracker, device, cfg, epoch, writer):
    model.eval(); tracker.reset()
    running, n = 0.0, 0
    pbar = tqdm(loader, desc=f"Ep {epoch:03d} VAL  ", leave=False, ncols=110)
    for batch in pbar:
        patches    = batch["patch"].to(device, non_blocking=True)
        cat_labels = {k: v.to(device) for k,v in batch["cat"].items()}
        reg_labels = {k: v.to(device) for k,v in batch["reg"].items()}
        with autocast(enabled=cfg["training"]["mixed_precision"]):
            out       = model(patches)
            loss, _   = loss_fn(out["cls"], out["reg"], cat_labels, reg_labels)
        tracker.update(out, batch)
        running += loss.item(); n += 1

    avg     = running / max(n, 1)
    metrics = tracker.compute()
    if writer:
        writer.add_scalar("Loss/val", avg, epoch)
        writer.add_scalar("Metrics/mean_OA",   metrics["mean_OA"],   epoch)
        writer.add_scalar("Metrics/mean_RMSE", metrics["mean_RMSE"], epoch)
        for k, v in metrics.items():
            writer.add_scalar(f"Val/{k}", v, epoch)
    return avg, metrics


# ──────────────────────────────────────────────────────────────
def main(args):
    cfg = load_cfg(args.config)
    p   = cfg["paths"]
    t   = cfg["training"]

    # ── Seed + Device ──────────────────────────────────────────
    set_seed(t["seed"])
    device = get_device(cfg)
    vram   = VRAMMonitor(device)

    # ── Load dataset files ─────────────────────────────────────
    print("\n" + "="*65)
    print("  Loading TAIGA dataset (memory-mapped — no full RAM load)")
    print("="*65)
    reader     = TAIGAEnviReader(p["hsi_file"], p["label_file"])
    bad_stands = load_bad_stands(p["bad_stands"])

    train_loader, val_loader, test_loader = get_dataloaders(reader, bad_stands, cfg)
    class_weights = train_loader.dataset.class_weights
    vram.print("After dataset init")

    # ── Model ──────────────────────────────────────────────────
    model = MultitaskModel(cfg).to(device)
    print(f"\n  Model parameters: {model.param_count():.2f} M")
    vram.print("After model init")

    # ── Loss + Optimiser ───────────────────────────────────────
    loss_fn = MultitaskLoss(class_weights, cfg).to(device)
    opt     = optim.Adam(list(model.parameters()) + list(loss_fn.parameters()),
                         lr=t["lr"], betas=(t["momentum"], 0.999),
                         weight_decay=t["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=t["num_epochs"]-t["warmup_epochs"], eta_min=1e-6)
    scaler    = GradScaler(enabled=t["mixed_precision"])

    # ── Resume ─────────────────────────────────────────────────
    start_epoch, best_rmse, history = 0, float("inf"), {
        "epoch":[], "train_loss":[], "val_loss":[], "mean_OA":[], "mean_RMSE":[]
    }
    if args.resume and os.path.exists(args.resume):
        start_epoch, best_rmse, history = load_ckpt(args.resume, model, opt, scaler)
        print(f"  Resumed from epoch {start_epoch}, best RMSE={best_rmse:.4f}")

    # ── Logging ────────────────────────────────────────────────
    os.makedirs(p["log_dir"],        exist_ok=True)
    os.makedirs(p["checkpoint_dir"], exist_ok=True)
    os.makedirs(p["results_dir"],    exist_ok=True)
    writer  = SummaryWriter(p["log_dir"])
    tracker = MetricsTracker()

    # ──────────────────────────────────────────────────────────
    print(f"\n  Starting training — {t['num_epochs']} epochs, "
          f"batch={t['batch_size']}, FP16={t['mixed_precision']}")
    print("="*65)

    for epoch in range(start_epoch + 1, t["num_epochs"] + 1):
        t0 = time.time()

        # Warmup LR
        if epoch <= t["warmup_epochs"]:
            frac = epoch / t["warmup_epochs"]
            for g in opt.param_groups:
                g["lr"] = t["lr"] * frac

        train_loss = train_one_epoch(model, train_loader, loss_fn,
                                     opt, scaler, device, cfg, epoch, writer)

        if epoch > t["warmup_epochs"]:
            scheduler.step()

        # ── Validate every N epochs ──
        if epoch % t["val_every"] == 0 or epoch == t["num_epochs"]:
            val_loss, metrics = validate(model, val_loader, loss_fn,
                                         tracker, device, cfg, epoch, writer)
            oa   = metrics["mean_OA"]
            rmse = metrics["mean_RMSE"]
            lr   = opt.param_groups[0]["lr"]
            print(f"  Ep {epoch:03d}/{t['num_epochs']} | "
                  f"Train={train_loss:.4f}  Val={val_loss:.4f} | "
                  f"OA={oa:.2f}%  RMSE={rmse:.4f} | "
                  f"LR={lr:.2e} | {time.time()-t0:.0f}s")

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["mean_OA"].append(oa)
            history["mean_RMSE"].append(rmse)

            if rmse < best_rmse:
                best_rmse = rmse
                save_ckpt(os.path.join(p["checkpoint_dir"],"best.pth"),
                          model, opt, scaler, epoch, best_rmse, history)
                print(f"  ✓ New best RMSE={best_rmse:.4f} — saved best.pth")

        if epoch % t["save_every"] == 0:
            save_ckpt(os.path.join(p["checkpoint_dir"],"latest.pth"),
                      model, opt, scaler, epoch, best_rmse, history)

        # VRAM check every 25 epochs
        if epoch % 25 == 0:
            vram.print(f"Epoch {epoch}")

    # ────────────────────────────────────────────────────────
    # Final test evaluation
    # ────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  Final TEST set evaluation")
    print("="*65)
    load_ckpt(os.path.join(p["checkpoint_dir"],"best.pth"), model, None, None)
    _, test_metrics = validate(model, test_loader, loss_fn,
                               tracker, device, cfg, 0, None)
    print(tracker.summary(test_metrics))

    # Save metrics JSON
    with open(os.path.join(p["results_dir"],"test_metrics.json"),"w") as f:
        json.dump({k: round(float(v),6) for k,v in test_metrics.items()}, f, indent=2)
    with open(os.path.join(p["results_dir"],"history.json"),"w") as f:
        json.dump(history, f, indent=2)

    # ────────────────────────────────────────────────────────
    # Generate prediction maps for presentation
    # ────────────────────────────────────────────────────────
    print("\n  Generating prediction maps (this may take 5–10 min)...")
    from visualization.prediction_maps import (
        predict_tile, get_gt_tile, plot_classification_maps,
        plot_regression_maps, plot_summary_dashboard, plot_training_curves
    )
    vc  = cfg["viz"]
    r0, r1 = vc["map_row_start"], vc["map_row_end"]
    c0, c1 = vc["map_col_start"], vc["map_col_end"]
    maps_dir = os.path.join(p["results_dir"], "maps")
    os.makedirs(maps_dir, exist_ok=True)

    model.eval()
    cls_pred, reg_pred, pred_mask = predict_tile(
        model, reader, r0, r1, c0, c1,
        patch_size=cfg["data"]["patch_size"],
        batch_size=128, device=device
    )
    cat_gt, reg_gt, gt_mask = get_gt_tile(reader, r0, r1, c0, c1)
    rgb  = reader.get_rgb(r0, r1, c0, c1)
    mask = pred_mask & gt_mask

    plot_classification_maps(rgb, cat_gt, cls_pred, mask,
        os.path.join(maps_dir,"classification_maps.png"), title="Test Region")
    plot_regression_maps(reg_gt, reg_pred, mask,
        os.path.join(maps_dir,"regression_maps.png"), title="Test Region")
    plot_summary_dashboard(rgb, cat_gt, cls_pred, reg_gt, reg_pred, mask,
        test_metrics, os.path.join(maps_dir,"summary_dashboard.png"),
        title=f"OA={test_metrics['mean_OA']:.1f}%  RMSE={test_metrics['mean_RMSE']:.4f}")
    plot_training_curves(history,
        os.path.join(maps_dir,"training_curves.png"))

    writer.close()
    print("\n✓ All done! Check results/ for maps and metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", default=None)
    main(parser.parse_args())
