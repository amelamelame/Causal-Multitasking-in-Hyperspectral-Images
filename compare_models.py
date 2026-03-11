"""
compare_models.py  —  Baseline vs Causal Model Comparison
===========================================================
Loads both trained checkpoints and generates:
  1. Side-by-side metrics table (all 13 tasks)
  2. Improvement bar charts
  3. Side-by-side prediction maps (RGB | GT | Baseline | Causal)
  4. Learned causal graph visualization
  5. Counterfactual explanation plots

Run:
  python compare_models.py
  python compare_models.py --baseline checkpoints/best.pth \
                           --causal   checkpoints/causal/best_causal.pth
"""

import os, json, argparse
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from torch.cuda.amp import autocast
from tqdm import tqdm

from data.envi_reader     import TAIGAEnviReader, CAT_VARIABLES, REG_VARIABLES, CLASS_NAMES
from data.taiga_dataset   import get_dataloaders, load_bad_stands
from models.baseline.model import MultitaskModel
from models.causal.causal_model import CausalMultitaskModel
from models.causal.counterfactual import CounterfactualExplainer
from losses.loss          import MultitaskLoss
from evaluation.metrics   import MetricsTracker
from utils.gpu_utils      import get_device
from visualization.prediction_maps import (
    predict_tile, get_gt_tile,
    CAT_COLORS, REG_CMAPS, REG_MAX_VAL, REG_UNITS
)


def evaluate_model(model, loader, device, cfg, label):
    """Run full test set evaluation. Returns metrics dict."""
    class_weights = loader.dataset.class_weights
    loss_fn = MultitaskLoss(class_weights, cfg).to(device)
    tracker = MetricsTracker()
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  Eval [{label}]", ncols=100):
            patches = batch["patch"].to(device)
            cat_lbl = {k: v.to(device) for k,v in batch["cat"].items()}
            reg_lbl = {k: v.to(device) for k,v in batch["reg"].items()}
            with autocast(enabled=cfg["training"]["mixed_precision"]):
                out = model(patches)
            tracker.update(out, batch)
    return tracker.compute()


@torch.no_grad()
def predict_tile_model(model, reader, r0, r1, c0, c1, ps, device, label):
    """Dense prediction over a tile for one model."""
    print(f"  Predicting tile [{label}]...")
    model.eval()
    H, W, half = r1-r0, c1-c0, ps//2
    cls_maps = {v: np.zeros((H,W), dtype=np.int8)    for v in CAT_VARIABLES}
    reg_maps = {v: np.zeros((H,W), dtype=np.float32) for v in REG_VARIABLES}
    mask     = np.zeros((H,W), dtype=bool)

    coords = [(r,c,r0+r,c0+c)
              for r in range(half, H-half)
              for c in range(half, W-half)
              if reader.is_valid_pixel(r0+r, c0+c)]

    for i in tqdm(range(0, len(coords), 256), desc=f"    {label}", ncols=90):
        bc   = coords[i:i+256]
        pts  = np.stack([reader.get_hsi_patch(ar,ac,ps) for _,_,ar,ac in bc])
        ptt  = torch.from_numpy(pts).float().to(device)
        out  = model(ptt)
        for j,(r,c,_,_) in enumerate(bc):
            for v in CAT_VARIABLES:
                l = out["cls"][v][j,:,ps//2,ps//2]
                cls_maps[v][r,c] = int(l.argmax())
            for v in REG_VARIABLES:
                reg_maps[v][r,c] = float(out["reg"][v][j,0,ps//2,ps//2])
            mask[r,c] = True
    return cls_maps, reg_maps, mask


# ─────────────────────────────────────────────────────────────
# Figure 1: Full metrics comparison table
# ─────────────────────────────────────────────────────────────

def plot_metrics_comparison(base_m, caus_m, save_path):
    """Side-by-side bar chart comparing all metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle("Baseline vs Causal Model — All Metrics", fontsize=16, fontweight='bold')

    # ── Classification OA comparison ──
    ax = axes[0]
    vars_c = list(CAT_VARIABLES.keys())
    base_oa = [base_m[f"{v}/OA"] for v in vars_c]
    caus_oa = [caus_m[f"{v}/OA"] for v in vars_c]
    base_f1 = [base_m[f"{v}/F1"] for v in vars_c]
    caus_f1 = [caus_m[f"{v}/F1"] for v in vars_c]

    x     = np.arange(len(vars_c))
    width = 0.2
    b1 = ax.bar(x-1.5*width, base_oa, width, label="Baseline OA",  color="#3498db", alpha=0.85)
    b2 = ax.bar(x-0.5*width, caus_oa, width, label="Causal OA",    color="#2ecc71", alpha=0.85)
    b3 = ax.bar(x+0.5*width, base_f1, width, label="Baseline F1",  color="#e74c3c", alpha=0.85)
    b4 = ax.bar(x+1.5*width, caus_f1, width, label="Causal F1",    color="#f39c12", alpha=0.85)

    # Add improvement arrows
    for i, (bo, co) in enumerate(zip(base_oa, caus_oa)):
        delta = co - bo
        color = "#27ae60" if delta >= 0 else "#e74c3c"
        ax.annotate(f"{delta:+.1f}%", xy=(x[i]-0.5*width, co+0.3),
                    ha='center', fontsize=8.5, color=color, fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels([v.replace("_"," ") for v in vars_c], fontsize=10)
    ax.set_ylabel("Score (%)"); ax.set_title("Classification Performance", fontsize=12)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3); ax.set_ylim(0, 105)

    # ── Regression RMSE comparison ──
    ax = axes[1]
    base_rm = [base_m[f"{v}/RMSE"] for v in REG_VARIABLES]
    caus_rm = [caus_m[f"{v}/RMSE"] for v in REG_VARIABLES]
    base_r2 = [base_m[f"{v}/R2"]   for v in REG_VARIABLES]
    caus_r2 = [caus_m[f"{v}/R2"]   for v in REG_VARIABLES]

    x     = np.arange(len(REG_VARIABLES))
    width = 0.35
    ax.bar(x-width/2, base_rm, width, label="Baseline RMSE", color="#3498db", alpha=0.85)
    ax.bar(x+width/2, caus_rm, width, label="Causal RMSE",   color="#2ecc71", alpha=0.85)

    for i, (br, cr) in enumerate(zip(base_rm, caus_rm)):
        delta = cr - br
        color = "#27ae60" if delta <= 0 else "#e74c3c"
        ax.annotate(f"{delta:+.4f}", xy=(x[i]+width/2, cr+0.001),
                    ha='center', fontsize=7, color=color, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("_","\n") for v in REG_VARIABLES], fontsize=7)
    ax.set_ylabel("RMSE (normalized)"); ax.set_title("Regression Performance", fontsize=12)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# Figure 2: Side-by-side prediction maps
# ─────────────────────────────────────────────────────────────

def plot_sidebyside_maps(rgb, cat_gt, cls_base, cls_caus,
                         reg_gt, reg_base, reg_caus, mask,
                         base_m, caus_m, save_path):
    """RGB | GT | Baseline | Causal — 4-column comparison."""
    cat_vars = list(CAT_VARIABLES.keys())
    sel_reg  = ["mean_height", "woody_biomass", "pct_pine", "leaf_area_index"]
    n_rows   = len(cat_vars) + len(sel_reg) + 1

    fig, axes = plt.subplots(n_rows, 4, figsize=(18, 3.8*n_rows))
    fig.suptitle("Prediction Map Comparison: Baseline vs Causal Model",
                 fontsize=16, fontweight='bold')

    col_headers = ["HSI Pseudo-RGB / Ground Truth", "Ground Truth",
                   "Baseline Prediction", "Causal Prediction"]
    for j, h in enumerate(col_headers):
        axes[0,j].set_title(h, fontsize=11, fontweight='bold', pad=8)

    # Row 0: RGB
    axes[0,0].imshow(rgb); axes[0,0].axis('off')
    for j in range(1,4): axes[0,j].axis('off')

    # Rows 1-3: Classification
    for i, v in enumerate(cat_vars):
        row = i + 1
        colors = CAT_COLORS[v]; n_cls = CAT_VARIABLES[v]
        cmap   = ListedColormap(['#cccccc'] + colors[:n_cls])
        bounds = [-0.5] + [c+0.5 for c in range(n_cls+1)]
        norm   = BoundaryNorm(bounds, cmap.N)
        cls_n  = CLASS_NAMES[v]

        for col_idx, (data, lbl) in enumerate([
            (cat_gt[v]+1,   "GT"),
            (cls_base[v]+1, f"Baseline  OA={base_m[f'{v}/OA']:.1f}%"),
            (cls_caus[v]+1, f"Causal    OA={caus_m[f'{v}/OA']:.1f}%"),
        ]):
            ax = axes[row, col_idx+1]
            ax.imshow(np.where(mask, data, 0), cmap=cmap, norm=norm,
                      interpolation='nearest')
            ax.set_title(f"{v}\n{lbl}", fontsize=9)
            ax.axis('off')

        axes[row,0].axis('off')
        patches = [mpatches.Patch(color=colors[c], label=cls_n.get(c+1,f"{c+1}"))
                   for c in range(n_cls)]
        axes[row,1].legend(handles=patches, fontsize=7, loc='lower right',
                           framealpha=0.8)

    # Rows 4-7: Regression
    for i, v in enumerate(sel_reg):
        row  = len(cat_vars) + i + 1
        cmap = REG_CMAPS[v]
        vmax = REG_MAX_VAL[v]

        for col_idx, (data, lbl) in enumerate([
            (reg_gt[v],   "Ground Truth"),
            (reg_base[v], f"Baseline RMSE={base_m[f'{v}/RMSE']:.3f}"),
            (reg_caus[v], f"Causal   RMSE={caus_m[f'{v}/RMSE']:.3f}"),
        ]):
            ax  = axes[row, col_idx+1]
            img = np.where(mask, data*vmax, np.nan)
            im  = ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax)
            ax.set_title(f"{v} [{REG_UNITS[v]}]\n{lbl}", fontsize=9)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.04)

        axes[row,0].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# Figure 3: Learned causal graph
# ─────────────────────────────────────────────────────────────

def plot_causal_graph(causal_model, save_path):
    """Visualize the learned causal adjacency matrix as a heatmap."""
    graph = causal_model.get_causal_graph()
    A     = graph["A_soft"]
    names = graph["var_names"]
    short = [n[:12] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Learned Causal Graph (Module 1 — NOTEARS DAG)",
                 fontsize=14, fontweight='bold')

    # Soft adjacency heatmap
    im0 = axes[0].imshow(A, cmap='YlOrRd', vmin=0, vmax=1)
    axes[0].set_xticks(range(len(names))); axes[0].set_xticklabels(short, rotation=45, ha='right', fontsize=8)
    axes[0].set_yticks(range(len(names))); axes[0].set_yticklabels(short, fontsize=8)
    axes[0].set_title("Soft Adjacency A[i→j]\n(row i causes column j)", fontsize=11)
    plt.colorbar(im0, ax=axes[0], fraction=0.04)

    # Hard adjacency (thresholded)
    A_hard = (A > 0.5).astype(float)
    axes[1].imshow(A_hard, cmap='Greens', vmin=0, vmax=1)
    axes[1].set_xticks(range(len(names))); axes[1].set_xticklabels(short, rotation=45, ha='right', fontsize=8)
    axes[1].set_yticks(range(len(names))); axes[1].set_yticklabels(short, fontsize=8)
    axes[1].set_title(f"Binary Graph (threshold=0.5)\nDAG penalty={graph['dag_pen']:.4f}", fontsize=11)

    # Annotate non-zero cells
    for i in range(len(names)):
        for j in range(len(names)):
            if A_hard[i,j] > 0:
                axes[1].text(j, i, "→", ha='center', va='center',
                             fontsize=9, color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# Figure 4: Counterfactual explanation
# ─────────────────────────────────────────────────────────────

def plot_cf_explanations(cf_explainer, test_loader, device, save_path):
    """Show spectral delta + wavelength region importances."""
    print("  Computing counterfactual explanations (sample 20 pixels)...")
    all_results = []
    n_target = 20

    for batch in test_loader:
        patches = batch["patch"]
        labels  = batch["cat"]["tree_species"]
        for i in range(len(patches)):
            if labels[i].item() == 0:   # pine → explain flip to spruce
                r = cf_explainer.explain(
                    patches[i:i+1], "tree_species", target_class=1, is_cls=True
                )
                all_results.append(r)
            if len(all_results) >= n_target:
                break
        if len(all_results) >= n_target:
            break

    if not all_results:
        print("  (No valid CF samples found)")
        return

    success_rate = np.mean([r["success"] for r in all_results])
    mean_delta   = np.mean([np.abs(r["delta"]) for r in all_results], axis=0)
    mean_import  = {}
    from visualization.prediction_maps import WAVELENGTH_REGIONS
    for k in WAVELENGTH_REGIONS:
        mean_import[k] = np.mean([r["region_import"][k] for r in all_results])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Counterfactual Explanations — Pine → Spruce  "
                 f"(success rate: {success_rate*100:.0f}%)",
                 fontsize=14, fontweight='bold')

    # Mean spectral delta across 128 bands
    bands = np.arange(128)
    wls   = np.linspace(400, 991, 128)
    axes[0].plot(wls, mean_delta, color='#e74c3c', linewidth=1.5)
    axes[0].fill_between(wls, 0, mean_delta, alpha=0.3, color='#e74c3c')
    axes[0].set_xlabel("Wavelength (nm)"); axes[0].set_ylabel("|Δ reflectance|")
    axes[0].set_title("Mean Spectral Perturbation\n(what must change to flip class)")
    axes[0].grid(alpha=0.3)
    # Shade wavelength regions
    from visualization.prediction_maps import WAVELENGTH_REGIONS
    colors_r = {"Blue_Green":"#3498db","Red":"#e74c3c",
                 "Red_edge":"#f39c12","NIR":"#8e44ad"}
    for name,(b0,b1) in WAVELENGTH_REGIONS.items():
        axes[0].axvspan(wls[b0], wls[min(b1,127)], alpha=0.07,
                        color=colors_r[name], label=name)
    axes[0].legend(fontsize=8)

    # Wavelength region importance bar
    regions = list(mean_import.keys())
    vals    = [mean_import[k] for k in regions]
    colors  = [colors_r[k] for k in regions]
    axes[1].bar(regions, vals, color=colors, alpha=0.85, edgecolor='black')
    axes[1].set_ylabel("Normalized Importance"); axes[1].set_ylim(0, 1)
    axes[1].set_title("Wavelength Region Importance\n(where model relies for decision)")
    for bar, v in zip(axes[1].patches, vals):
        axes[1].text(bar.get_x()+bar.get_width()/2, v+0.01,
                     f"{v:.2f}", ha='center', fontsize=11, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    # Success/failure pie
    s = int(success_rate * len(all_results))
    f = len(all_results) - s
    axes[2].pie([s,f], labels=[f"Success\n({s})", f"Failed\n({f})"],
                colors=["#2ecc71","#e74c3c"], autopct="%1.0f%%",
                startangle=90, textprops={"fontsize":12})
    axes[2].set_title(f"CF Flip Success Rate\n{len(all_results)} samples, Pine→Spruce")

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# Summary table (printed + saved as image)
# ─────────────────────────────────────────────────────────────

def print_comparison_table(base_m, caus_m):
    print("\n" + "="*80)
    print(f"  {'METRIC':<35} {'BASELINE':>12}  {'CAUSAL':>12}  {'ΔIMPROVE':>10}")
    print("─"*80)
    print("  CLASSIFICATION:")
    for v in CAT_VARIABLES:
        for metric in ["OA","F1","AUC"]:
            bv = base_m.get(f"{v}/{metric}", 0)
            cv = caus_m.get(f"{v}/{metric}", 0)
            d  = cv - bv
            sym = "↑" if d>0 else "↓"
            print(f"  {v}/{metric:<31} {bv:>11.2f}%  {cv:>11.2f}%  {sym}{abs(d):>8.2f}%")
    print("─"*80)
    print("  REGRESSION (RMSE — lower is better):")
    for v in REG_VARIABLES:
        bv = base_m.get(f"{v}/RMSE", 0)
        cv = caus_m.get(f"{v}/RMSE", 0)
        d  = cv - bv
        sym = "↓" if d<=0 else "↑"
        print(f"  {v}/RMSE{' '*27} {bv:>12.4f}  {cv:>12.4f}  {sym}{abs(d):>9.4f}")
    print("─"*80)
    print(f"  {'Mean OA':<35} {base_m['mean_OA']:>11.2f}%  {caus_m['mean_OA']:>11.2f}%"
          f"  {'↑' if caus_m['mean_OA']>base_m['mean_OA'] else '↓'}"
          f"{abs(caus_m['mean_OA']-base_m['mean_OA']):>8.2f}%")
    print(f"  {'Mean RMSE':<35} {base_m['mean_RMSE']:>12.4f}  {caus_m['mean_RMSE']:>12.4f}"
          f"  {'↓' if caus_m['mean_RMSE']<base_m['mean_RMSE'] else '↑'}"
          f"{abs(caus_m['mean_RMSE']-base_m['mean_RMSE']):>9.4f}")
    print("="*80)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main(args):
    cfg_b = yaml.safe_load(open(args.baseline_cfg))
    cfg_c = yaml.safe_load(open(args.causal_cfg))
    device = get_device(cfg_b)

    out_dir = "results/comparison"
    os.makedirs(out_dir, exist_ok=True)

    p_b = cfg_b["paths"]; p_c = cfg_c["paths"]

    # ── Load data ──────────────────────────────────────────────
    print("\nLoading TAIGA (memory-mapped)...")
    reader     = TAIGAEnviReader(p_b["hsi_file"], p_b["label_file"])
    bad_stands = load_bad_stands(p_b["bad_stands"])
    _, _, test_ldr = get_dataloaders(reader, bad_stands, cfg_b)

    # ── Load models ────────────────────────────────────────────
    print("\nLoading baseline model...")
    base_model = MultitaskModel(cfg_b).to(device)
    ck_b = torch.load(args.baseline, map_location=device)
    base_model.load_state_dict(ck_b["model"])

    print("Loading causal model...")
    caus_model = CausalMultitaskModel(cfg_c).to(device)
    ck_c = torch.load(args.causal, map_location=device)
    caus_model.load_state_dict(ck_c["model"])

    # ── Evaluate both ──────────────────────────────────────────
    print("\nEvaluating baseline...")
    base_m = evaluate_model(base_model, test_ldr, device, cfg_b, "Baseline")
    print("Evaluating causal...")
    caus_m = evaluate_model(caus_model, test_ldr, device, cfg_c, "Causal")

    # Save both metrics
    with open(os.path.join(out_dir,"metrics_baseline.json"),"w") as f:
        json.dump({k:round(float(v),6) for k,v in base_m.items()}, f, indent=2)
    with open(os.path.join(out_dir,"metrics_causal.json"),"w") as f:
        json.dump({k:round(float(v),6) for k,v in caus_m.items()}, f, indent=2)

    # Print table
    print_comparison_table(base_m, caus_m)

    # ── Generate all figures ───────────────────────────────────
    vc   = cfg_b["viz"]
    r0,r1 = vc["map_row_start"],vc["map_row_end"]
    c0,c1 = vc["map_col_start"],vc["map_col_end"]
    ps    = cfg_b["data"]["patch_size"]

    print("\nGenerating prediction maps (both models)...")
    cls_b, reg_b, mask_b = predict_tile_model(base_model, reader, r0,r1,c0,c1, ps, device,"Baseline")
    cls_c, reg_c, mask_c = predict_tile_model(caus_model, reader, r0,r1,c0,c1, ps, device,"Causal")
    cat_gt, reg_gt, gt_mask = get_gt_tile(reader, r0,r1,c0,c1)
    rgb  = reader.get_rgb(r0,r1,c0,c1)
    mask = mask_b & mask_c & gt_mask

    print("\nPlotting figures...")
    plot_metrics_comparison(base_m, caus_m,
        os.path.join(out_dir,"1_metrics_comparison.png"))
    plot_sidebyside_maps(rgb, cat_gt, cls_b, cls_c, reg_gt, reg_b, reg_c,
        mask, base_m, caus_m,
        os.path.join(out_dir,"2_sidebyside_maps.png"))
    plot_causal_graph(caus_model,
        os.path.join(out_dir,"3_causal_graph.png"))

    # CF explanations
    cf_exp = CounterfactualExplainer(caus_model, device,
                max_iter   = cfg_c["causal"]["cf_max_iter"],
                lambda_dist = cfg_c["causal"]["cf_lambda_dist"])
    plot_cf_explanations(cf_exp, test_ldr, device,
        os.path.join(out_dir,"4_cf_explanations.png"))

    print(f"\n✓ Comparison complete!  All figures in {out_dir}/")
    print("  1_metrics_comparison.png   ← bar chart of all metrics")
    print("  2_sidebyside_maps.png      ← RGB|GT|Baseline|Causal maps")
    print("  3_causal_graph.png         ← learned causal DAG heatmap")
    print("  4_cf_explanations.png      ← spectral CF importance")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_cfg", default="configs/config.yaml")
    ap.add_argument("--causal_cfg",   default="configs/config_causal.yaml")
    ap.add_argument("--baseline",     default="checkpoints/best.pth")
    ap.add_argument("--causal",       default="checkpoints/causal/best_causal.pth")
    main(ap.parse_args())
