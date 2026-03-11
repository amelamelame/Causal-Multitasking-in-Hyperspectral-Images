"""
train_causal.py  —  Train the Full Causal Model
=================================================
Implements the full annealing schedule:
  Epochs 1–9:   Baseline losses only (warm up encoder)
  Epochs 10–19: + DAG constraint (causal graph starts forming)
  Epochs 20–150: + IRM + HSIC + full causal training

Usage:
  python train_causal.py
  python train_causal.py --resume checkpoints/causal/latest.pth
  python train_causal.py --init_from_baseline checkpoints/best.pth
"""

import os, sys, time, argparse, json
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from data.envi_reader     import TAIGAEnviReader, CAT_VARIABLES, REG_VARIABLES
from data.taiga_dataset   import get_dataloaders, load_bad_stands
from models.causal.causal_model  import CausalMultitaskModel
from models.causal.counterfactual import CounterfactualAugmentor
from losses.loss          import MultitaskLoss
from losses.causal_loss   import CausalLoss
from evaluation.metrics   import MetricsTracker
from utils.gpu_utils      import set_seed, get_device, VRAMMonitor


def load_cfg(path):
    with open(path) as f: return yaml.safe_load(f)


def save_ckpt(path, model, opt, scaler, epoch, best, history):
    torch.save({"epoch":epoch,"model":model.state_dict(),
                "opt":opt.state_dict(),"scaler":scaler.state_dict(),
                "best":best,"history":history}, path)


def load_ckpt(path, model, opt=None, scaler=None):
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model"], strict=False)
    if opt and "opt" in ck:    opt.load_state_dict(ck["opt"])
    if scaler and "scaler" in ck: scaler.load_state_dict(ck["scaler"])
    return ck.get("epoch",0), ck.get("best",float("inf")), ck.get("history",{
        "epoch":[],"train_loss":[],"val_loss":[],"mean_OA":[],"mean_RMSE":[],
        "L_irm":[],"L_dag":[],"L_hsic":[]})


def train_one_epoch(model, loader, loss_fn, opt, scaler, device, cfg, epoch,
                    cf_aug, writer):
    model.train()
    running, n = 0.0, 0
    det_accum = {}
    pbar = tqdm(loader, desc=f"Ep{epoch:03d} CAUSAL", leave=False, ncols=115)

    for batch in pbar:
        # CF augmentation for minority classes (Module 4b, epoch >= 20)
        if epoch >= 20 and cf_aug is not None and np.random.random() < 0.3:
            try:
                # Augment soil_class (often imbalanced) toward organic peat (cls 1)
                batch = cf_aug.get_augmented_batch(batch, "soil_class", target_cls=1)
            except Exception:
                pass  # skip if augmentation fails (model not warm enough)

        patches    = batch["patch"].to(device, non_blocking=True)
        cat_labels = {k: v.to(device) for k,v in batch["cat"].items()}
        reg_labels = {k: v.to(device) for k,v in batch["reg"].items()}
        env_ids    = batch.get("env_id", torch.zeros(patches.shape[0],dtype=torch.long)).to(device)

        opt.zero_grad(set_to_none=True)
        with autocast(enabled=cfg["training"]["mixed_precision"]):
            out       = model(patches, env_ids)
            loss, det = loss_fn(out, cat_labels, reg_labels, env_ids, epoch)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
        scaler.step(opt); scaler.update()

        running += loss.item(); n += 1
        for k,v in det.items():
            det_accum[k] = det_accum.get(k,0) + v
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         dag=f"{det.get('L_dag',0):.3f}",
                         irm=f"{det.get('L_irm',0):.3f}")

    avg = running / max(n,1)
    avg_det = {k:v/max(n,1) for k,v in det_accum.items()}
    if writer:
        writer.add_scalar("Loss/train_causal", avg, epoch)
        for k,v in avg_det.items():
            writer.add_scalar(f"CausalLoss/{k}", v, epoch)
    return avg, avg_det


@torch.no_grad()
def validate(model, loader, loss_fn, tracker, device, cfg, epoch, writer):
    model.eval(); tracker.reset()
    running, n = 0.0, 0
    for batch in tqdm(loader, desc=f"Ep{epoch:03d} VAL  ", leave=False, ncols=115):
        patches    = batch["patch"].to(device)
        cat_labels = {k: v.to(device) for k,v in batch["cat"].items()}
        reg_labels = {k: v.to(device) for k,v in batch["reg"].items()}
        env_ids    = batch.get("env_id", torch.zeros(patches.shape[0],dtype=torch.long)).to(device)
        with autocast(enabled=cfg["training"]["mixed_precision"]):
            out     = model(patches, env_ids)
            loss,_  = loss_fn(out, cat_labels, reg_labels, env_ids, 999)
        tracker.update(out, batch)
        running += loss.item(); n += 1

    avg     = running / max(n,1)
    metrics = tracker.compute()
    if writer:
        writer.add_scalar("Loss/val_causal", avg, epoch)
        writer.add_scalar("Metrics/mean_OA",   metrics["mean_OA"],   epoch)
        writer.add_scalar("Metrics/mean_RMSE", metrics["mean_RMSE"], epoch)
    return avg, metrics


def main(args):
    cfg    = load_cfg(args.config)
    p      = cfg["paths"]
    t      = cfg["training"]

    set_seed(t["seed"])
    device = get_device(cfg)
    vram   = VRAMMonitor(device)

    print("\n" + "="*65)
    print("  CAUSAL Multitask Model — Training")
    print("  Modules: Disentangle | DAG | Causal Decoders | CF Aug")
    print("="*65)

    # ── Data ───────────────────────────────────────────────────
    reader     = TAIGAEnviReader(p["hsi_file"], p["label_file"])
    bad_stands = load_bad_stands(p["bad_stands"])
    train_ldr, val_ldr, test_ldr = get_dataloaders(reader, bad_stands, cfg)
    class_weights = train_ldr.dataset.class_weights
    vram.print("After dataset")

    # ── Model ──────────────────────────────────────────────────
    model = CausalMultitaskModel(cfg).to(device)
    print(f"  Causal model: {model.param_count():.2f} M parameters")
    vram.print("After model init")

    # ── Optionally init encoder from trained baseline ──────────
    if args.init_from_baseline and os.path.exists(args.init_from_baseline):
        ck = torch.load(args.init_from_baseline, map_location="cpu")
        # Load only matching keys (encoder weights)
        new_state = model.state_dict()
        matched   = {k: v for k, v in ck["model"].items() if k in new_state
                     and new_state[k].shape == v.shape}
        new_state.update(matched)
        model.load_state_dict(new_state)
        print(f"  ✓ Initialized from baseline: {len(matched)} layers transferred")

    # ── Loss functions ─────────────────────────────────────────
    base_loss   = MultitaskLoss(class_weights, cfg).to(device)
    causal_loss = CausalLoss(base_loss, n_envs=cfg["causal"]["num_envs"],
                             cfg=cfg).to(device)

    # ── Optimiser ──────────────────────────────────────────────
    opt = optim.Adam(
        list(model.parameters()) + list(causal_loss.parameters()),
        lr=t["lr"], betas=(t["momentum"], 0.999), weight_decay=t["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=t["num_epochs"]-t["warmup_epochs"], eta_min=1e-6)
    scaler    = GradScaler(enabled=t["mixed_precision"])

    # ── CF Augmentor (Module 4b) ────────────────────────────────
    cf_aug = CounterfactualAugmentor(
        model, device,
        step_size = cfg["causal"]["cf_step_size"],
        n_steps   = cfg["causal"]["cf_n_steps"],
    )

    # ── Resume ─────────────────────────────────────────────────
    hist_keys = ["epoch","train_loss","val_loss","mean_OA","mean_RMSE",
                 "L_irm","L_dag","L_hsic"]
    start_epoch, best_rmse, history = 0, float("inf"), {k:[] for k in hist_keys}
    if args.resume and os.path.exists(args.resume):
        start_epoch, best_rmse, history = load_ckpt(args.resume, model, opt, scaler)
        print(f"  Resumed from epoch {start_epoch}, best={best_rmse:.4f}")

    # ── Logging ────────────────────────────────────────────────
    os.makedirs(p["checkpoint_dir"], exist_ok=True)
    os.makedirs(p["results_dir"],    exist_ok=True)
    os.makedirs(p["log_dir"],        exist_ok=True)
    writer  = SummaryWriter(p["log_dir"])
    tracker = MetricsTracker()

    print(f"\n  Epochs: {t['num_epochs']}  | Batch: {t['batch_size']} | FP16: {t['mixed_precision']}")
    print(f"  DAG loss starts:  epoch {cfg['causal']['dag_anneal_epoch']}")
    print(f"  IRM+HSIC starts:  epoch {cfg['causal']['irm_anneal_epoch']}")
    print("="*65)

    for epoch in range(start_epoch + 1, t["num_epochs"] + 1):
        t0 = time.time()

        # Phase indicator
        if epoch < cfg["causal"]["dag_anneal_epoch"]:
            phase = "WARMUP"
        elif epoch < cfg["causal"]["irm_anneal_epoch"]:
            phase = "DAG"
        else:
            phase = "FULL-CAUSAL"

        # Warmup LR
        if epoch <= t["warmup_epochs"]:
            for g in opt.param_groups:
                g["lr"] = t["lr"] * (epoch / t["warmup_epochs"])

        tr_loss, tr_det = train_one_epoch(
            model, train_ldr, causal_loss, opt, scaler, device, cfg, epoch,
            cf_aug, writer)

        if epoch > t["warmup_epochs"]:
            scheduler.step()

        if epoch % t["val_every"] == 0 or epoch == t["num_epochs"]:
            vl, mets = validate(model, val_ldr, causal_loss, tracker,
                                device, cfg, epoch, writer)
            oa   = mets["mean_OA"]
            rmse = mets["mean_RMSE"]
            lr   = opt.param_groups[0]["lr"]
            print(f"  [{phase:12s}] Ep{epoch:03d} | "
                  f"Tr={tr_loss:.4f} Val={vl:.4f} | "
                  f"OA={oa:.2f}% RMSE={rmse:.4f} | "
                  f"DAG={tr_det.get('L_dag',0):.3f} IRM={tr_det.get('L_irm',0):.4f} | "
                  f"LR={lr:.2e} {time.time()-t0:.0f}s")

            for k in hist_keys[1:]:
                history[k].append(
                    tr_det.get(k, mets.get(k, tr_loss if k=="train_loss" else vl)))
            history["epoch"].append(epoch)
            history["train_loss"][-1] = tr_loss
            history["val_loss"][-1]   = vl
            history["mean_OA"][-1]    = oa
            history["mean_RMSE"][-1]  = rmse

            if rmse < best_rmse:
                best_rmse = rmse
                save_ckpt(os.path.join(p["checkpoint_dir"],"best_causal.pth"),
                          model, opt, scaler, epoch, best_rmse, history)
                print(f"  ✓ Best RMSE={best_rmse:.4f} — saved best_causal.pth")

            # Print causal graph every 25 epochs
            if epoch % 25 == 0:
                model.dag.print_graph()

        if epoch % t["save_every"] == 0:
            save_ckpt(os.path.join(p["checkpoint_dir"],"latest_causal.pth"),
                      model, opt, scaler, epoch, best_rmse, history)

        if epoch % 25 == 0:
            vram.print(f"Epoch {epoch}")

    # ── Final test + maps ──────────────────────────────────────
    print("\n" + "="*65 + "\n  Final TEST evaluation\n" + "="*65)
    load_ckpt(os.path.join(p["checkpoint_dir"],"best_causal.pth"), model)
    _, test_mets = validate(model, test_ldr, causal_loss, tracker, device, cfg, 0, None)
    print(tracker.summary(test_mets))

    with open(os.path.join(p["results_dir"],"test_metrics_causal.json"),"w") as f:
        json.dump({k:round(float(v),6) for k,v in test_mets.items()}, f, indent=2)
    with open(os.path.join(p["results_dir"],"history_causal.json"),"w") as f:
        json.dump(history, f, indent=2)

    # Print learned causal graph
    model.dag.print_graph()

    writer.close()
    print("\n✓ Causal training complete!")
    print(f"  Run comparison: python compare_models.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",             default="configs/config_causal.yaml")
    ap.add_argument("--resume",             default=None)
    ap.add_argument("--init_from_baseline", default="checkpoints/best.pth",
                    help="Initialize encoder from trained baseline checkpoint")
    main(ap.parse_args())
