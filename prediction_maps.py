"""
visualization/prediction_maps.py
==================================
Generates colorful prediction maps for presenting results to your professor.
Creates a figure comparing:
  - RGB pseudo-color image (from HSI bands 97, 65, 33)
  - Ground truth maps for all 13 variables
  - Predicted maps for all 13 variables
  - Error/difference maps
  - Confusion matrices
  - Scatter plots for regression
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, ListedColormap
import matplotlib.gridspec as gridspec
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

from data.envi_reader import (
    CAT_VARIABLES, REG_VARIABLES, CLASS_NAMES, REG_VARIABLES
)


# ─────────────────────────────────────────────────────────────
# Color palettes
# ─────────────────────────────────────────────────────────────

# Categorical variable colormaps
CAT_COLORS = {
    "fertility_class": ["#2ecc71","#27ae60","#f39c12","#e67e22","#e74c3c","#c0392b"],
    "soil_class":      ["#3498db","#8e44ad"],
    "tree_species":    ["#e74c3c","#2980b9","#27ae60"],
}

# Regression variable colormaps (sequential, physically meaningful)
REG_CMAPS = {
    "basal_area":         "YlGn",
    "mean_dbh":           "YlOrBr",
    "stem_density":       "Blues",
    "mean_height":        "Greens",
    "pct_pine":           "Reds",
    "pct_spruce":         "Blues",
    "pct_birch":          "BuGn",
    "woody_biomass":      "YlOrRd",
    "leaf_area_index":    "RdYlGn",
    "eff_leaf_area_index":"RdYlGn",
}

REG_UNITS = {
    "basal_area": "m²/ha", "mean_dbh": "cm",
    "stem_density": "1/ha", "mean_height": "m",
    "pct_pine": "%", "pct_spruce": "%", "pct_birch": "%",
    "woody_biomass": "t/ha", "leaf_area_index": "-",
    "eff_leaf_area_index": "-",
}

REG_MAX_VAL = {
    "basal_area":35.51,"mean_dbh":30.89,"stem_density":6240,"mean_height":24.16,
    "pct_pine":100,"pct_spruce":84,"pct_birch":58,
    "woody_biomass":180,"leaf_area_index":9.66,"eff_leaf_area_index":6.45,
}


# ─────────────────────────────────────────────────────────────
# Dense prediction over a tile
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_tile(model, reader, row0, row1, col0, col1,
                 patch_size=9, batch_size=256, device="cuda"):
    """
    Run dense pixel-wise prediction over a spatial tile.
    Returns:
      cls_maps: dict[var -> H×W int array]
      reg_maps: dict[var -> H×W float array, normalized [0,1]]
    """
    model.eval()
    H = row1 - row0
    W = col1 - col0
    half = patch_size // 2

    # Pre-allocate output arrays
    cls_maps = {v: np.zeros((H,W), dtype=np.int8)   for v in CAT_VARIABLES}
    reg_maps = {v: np.zeros((H,W), dtype=np.float32) for v in REG_VARIABLES}
    mask     = np.zeros((H,W), dtype=bool)

    # Collect valid pixel coordinates in tile
    coords = []
    for r in range(half, H-half):
        for c in range(half, W-half):
            abs_r = row0 + r
            abs_c = col0 + c
            if reader.is_valid_pixel(abs_r, abs_c):
                coords.append((r, c, abs_r, abs_c))

    print(f"  Predicting {len(coords):,} valid pixels in tile {H}×{W}...")

    # Process in batches
    for i in tqdm(range(0, len(coords), batch_size), desc="  Predicting"):
        batch_coords = coords[i:i+batch_size]
        patches = np.stack([
            reader.get_hsi_patch(ar, ac, patch_size)
            for _, _, ar, ac in batch_coords
        ])   # [B, 128, 9, 9]
        patches_t = torch.from_numpy(patches).float().to(device)

        out = model(patches_t)

        for j, (r, c, ar, ac) in enumerate(batch_coords):
            for v in CAT_VARIABLES:
                n = CAT_VARIABLES[v]
                logits = out["cls"][v][j, :, patch_size//2, patch_size//2]
                cls_maps[v][r, c] = int(logits.argmax().item())
            for v in REG_VARIABLES:
                reg_maps[v][r, c] = float(out["reg"][v][j, 0, patch_size//2, patch_size//2])
            mask[r, c] = True

    return cls_maps, reg_maps, mask


def get_gt_tile(reader, row0, row1, col0, col1):
    """Extract ground truth maps for a tile."""
    H, W = row1-row0, col1-col0
    lbl  = reader.get_label_tile(row0, row1, col0, col1)  # [H,W,14]

    from data.envi_reader import LABEL_SCALE
    band_map = {
        "basal_area":3,"mean_dbh":4,"stem_density":5,"mean_height":6,
        "pct_pine":7,"pct_spruce":8,"pct_birch":9,
        "woody_biomass":10,"leaf_area_index":11,"eff_leaf_area_index":12,
    }
    cat_gt = {
        "fertility_class": np.clip(lbl[:,:,0]-1, 0, 5).astype(np.int8),
        "soil_class":      np.clip(lbl[:,:,1]-1, 0, 1).astype(np.int8),
        "tree_species":    np.clip(lbl[:,:,2]-1, 0, 2).astype(np.int8),
    }
    reg_gt = {}
    for v in REG_VARIABLES:
        bi  = band_map[v]
        arr = lbl[:,:,bi].astype(np.float32)
        if v in LABEL_SCALE:
            arr /= LABEL_SCALE[v]
        arr = np.clip(arr / (REG_MAX_VAL[v]+1e-8), 0, 1)
        reg_gt[v] = arr

    valid = (lbl[:,:,0] > 0) & (lbl[:,:,1] > 0) & (lbl[:,:,2] > 0)
    return cat_gt, reg_gt, valid.astype(bool)


# ─────────────────────────────────────────────────────────────
# Figure 1: Overview map (RGB + all classification results)
# ─────────────────────────────────────────────────────────────

def plot_classification_maps(rgb, cat_gt, cat_pred, mask, save_path, title=""):
    """
    Big figure: RGB | GT | Pred for each categorical variable.
    """
    vars_list = list(CAT_VARIABLES.keys())
    n_vars    = len(vars_list)
    fig, axes = plt.subplots(n_vars+1, 3, figsize=(15, 4*(n_vars+1)))
    fig.suptitle(f"Classification Results — {title}", fontsize=16, fontweight='bold', y=1.01)

    # Row 0: RGB overview
    axes[0,0].imshow(rgb); axes[0,0].set_title("HSI Pseudo-RGB\n(bands 97,65,33)",fontsize=11)
    axes[0,0].axis('off')
    axes[0,1].axis('off'); axes[0,2].axis('off')

    for i, v in enumerate(vars_list):
        row = i + 1
        colors = CAT_COLORS[v]
        n_cls  = CAT_VARIABLES[v]
        cmap   = ListedColormap(['#cccccc'] + colors[:n_cls])
        bounds = [-0.5] + [c+0.5 for c in range(n_cls+1)]
        norm   = BoundaryNorm(bounds, cmap.N)
        cls_names = CLASS_NAMES[v]

        # GT
        gt_img = np.where(mask, cat_gt[v]+1, 0)
        axes[row,0].imshow(gt_img, cmap=cmap, norm=norm, interpolation='nearest')
        axes[row,0].set_title(f"{v}\nGround Truth", fontsize=10)
        axes[row,0].axis('off')

        # Prediction
        pr_img = np.where(mask, cat_pred[v]+1, 0)
        axes[row,1].imshow(pr_img, cmap=cmap, norm=norm, interpolation='nearest')
        axes[row,1].set_title(f"{v}\nPrediction", fontsize=10)
        axes[row,1].axis('off')

        # Error map (correct=green, wrong=red)
        correct = (cat_gt[v] == cat_pred[v]) & mask
        err_img = np.zeros((*mask.shape, 3), dtype=np.uint8)
        err_img[correct]        = [46, 204, 113]   # green = correct
        err_img[mask & ~correct]= [231, 76, 60]    # red   = wrong
        err_img[~mask]          = [200, 200, 200]   # grey  = no data
        axes[row,2].imshow(err_img)
        acc = correct.sum() / (mask.sum()+1e-8) * 100
        axes[row,2].set_title(f"Correct/Incorrect\nOA = {acc:.1f}%", fontsize=10)
        axes[row,2].axis('off')

        # Legend patches
        patches = [mpatches.Patch(color=colors[c], label=cls_names.get(c+1,f"cls{c+1}"))
                   for c in range(n_cls)]
        axes[row,1].legend(handles=patches, loc='lower right', fontsize=7,
                           framealpha=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# Figure 2: Regression maps
# ─────────────────────────────────────────────────────────────

def plot_regression_maps(reg_gt, reg_pred, mask, save_path, title=""):
    """
    Grid: GT | Prediction | Error for each continuous variable.
    """
    n = len(REG_VARIABLES)
    fig, axes = plt.subplots(n, 3, figsize=(15, 3.5*n))
    fig.suptitle(f"Regression Results — {title}", fontsize=16, fontweight='bold')

    for i, v in enumerate(REG_VARIABLES):
        cmap = REG_CMAPS[v]
        gt  = np.where(mask, reg_gt[v], np.nan)
        pr  = np.where(mask, reg_pred[v], np.nan)

        # Real-world scale for colorbar
        vmax_real = REG_MAX_VAL[v]

        im0 = axes[i,0].imshow(gt*vmax_real, cmap=cmap, vmin=0, vmax=vmax_real)
        axes[i,0].set_title(f"{v}\nGround Truth [{REG_UNITS[v]}]", fontsize=9)
        axes[i,0].axis('off')
        plt.colorbar(im0, ax=axes[i,0], fraction=0.04)

        im1 = axes[i,1].imshow(pr*vmax_real, cmap=cmap, vmin=0, vmax=vmax_real)
        axes[i,1].set_title(f"{v}\nPrediction [{REG_UNITS[v]}]", fontsize=9)
        axes[i,1].axis('off')
        plt.colorbar(im1, ax=axes[i,1], fraction=0.04)

        # Absolute error map
        err = np.abs(gt - pr) * vmax_real
        im2 = axes[i,2].imshow(err, cmap='Reds', vmin=0, vmax=vmax_real*0.3)
        valid_mask = ~np.isnan(err)
        mae  = float(np.nanmean(err))
        axes[i,2].set_title(f"Abs Error  MAE={mae:.2f} {REG_UNITS[v]}", fontsize=9)
        axes[i,2].axis('off')
        plt.colorbar(im2, ax=axes[i,2], fraction=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# Figure 3: Summary dashboard for presentation
# ─────────────────────────────────────────────────────────────

def plot_summary_dashboard(rgb, cat_gt, cat_pred, reg_gt, reg_pred,
                           mask, metrics, save_path, title=""):
    """
    Single-page dashboard: best results for professor presentation.
    Shows: RGB, 3 classification maps, 4 regression maps, metrics table.
    """
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f"Multitask Deep Learning Results — TAIGA Dataset\n{title}",
                 fontsize=17, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.35, wspace=0.3)

    # ── Row 0: RGB + 3 classification predictions ──
    ax_rgb = fig.add_subplot(gs[0,0])
    ax_rgb.imshow(rgb); ax_rgb.set_title("HSI Pseudo-RGB", fontsize=11, fontweight='bold')
    ax_rgb.axis('off')

    cat_vars = list(CAT_VARIABLES.keys())
    for j, v in enumerate(cat_vars):
        ax = fig.add_subplot(gs[0, j+1])
        colors = CAT_COLORS[v]
        n_cls  = CAT_VARIABLES[v]
        cmap   = ListedColormap(['#cccccc'] + colors[:n_cls])
        bounds = [-0.5] + [c+0.5 for c in range(n_cls+1)]
        norm   = BoundaryNorm(bounds, cmap.N)
        pr_img = np.where(mask, cat_pred[v]+1, 0)
        ax.imshow(pr_img, cmap=cmap, norm=norm, interpolation='nearest')
        oa  = metrics.get(f"{v}/OA", 0)
        ax.set_title(f"{v}\nOA={oa:.1f}%", fontsize=10, fontweight='bold')
        ax.axis('off')
        # Mini legend
        cls_names = CLASS_NAMES[v]
        patches = [mpatches.Patch(color=colors[c], label=cls_names.get(c+1,f"{c+1}"))
                   for c in range(n_cls)]
        ax.legend(handles=patches, fontsize=6.5, loc='lower right', framealpha=0.7)

    # Blank 5th column row 0
    fig.add_subplot(gs[0,4]).axis('off')

    # ── Row 1-2: 4 regression variable predictions (2 rows × 4 cols) ──
    sel_vars = ["mean_height","woody_biomass","pct_pine","leaf_area_index",
                "basal_area","stem_density","pct_spruce","pct_birch"]
    for idx, v in enumerate(sel_vars):
        row = 1 + idx // 4
        col = idx % 4
        ax  = fig.add_subplot(gs[row, col])
        cmap = REG_CMAPS[v]
        pr   = np.where(mask, reg_pred[v] * REG_MAX_VAL[v], np.nan)
        vmax = REG_MAX_VAL[v]
        im   = ax.imshow(pr, cmap=cmap, vmin=0, vmax=vmax)
        rmse = metrics.get(f"{v}/RMSE", 0) * vmax
        ax.set_title(f"{v}\nRMSE={rmse:.2f} {REG_UNITS[v]}", fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.05, pad=0.02)

    # ── Metrics summary text box ──
    ax_txt = fig.add_subplot(gs[1:, 4])
    ax_txt.axis('off')
    lines  = ["METRICS SUMMARY\n"]
    lines += ["Classification:"]
    for v in CAT_VARIABLES:
        oa = metrics.get(f"{v}/OA", 0)
        f1 = metrics.get(f"{v}/F1", 0)
        lines.append(f"  {v[:18]:<18} OA={oa:.1f}%  F1={f1:.1f}%")
    lines += ["\nRegression (mean):"]
    for v in REG_VARIABLES:
        rmse = metrics.get(f"{v}/RMSE", 0)
        r2   = metrics.get(f"{v}/R2", 0)
        lines.append(f"  {v[:18]:<18} RMSE={rmse:.3f}  R²={r2:.3f}")
    lines += [f"\nMean OA:   {metrics.get('mean_OA',0):.2f}%"]
    lines += [f"Mean RMSE: {metrics.get('mean_RMSE',0):.4f}"]

    ax_txt.text(0.02, 0.97, "\n".join(lines), transform=ax_txt.transAxes,
                fontsize=8.5, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f0f4ff', alpha=0.9))

    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# Figure 4: Training curves
# ─────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, save_path: str):
    """Plot loss curves and key metric curves from training history."""
    epochs   = history["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training Curves — TAIGA Multitask Model", fontsize=14, fontweight='bold')

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#2980b9")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="#e74c3c")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Multitask Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # Mean OA
    axes[1].plot(epochs, history["mean_OA"], color="#27ae60", marker='o', ms=3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("OA (%)")
    axes[1].set_title("Mean Classification OA"); axes[1].grid(alpha=0.3)

    # Mean RMSE
    axes[2].plot(epochs, history["mean_RMSE"], color="#8e44ad", marker='o', ms=3)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("RMSE (normalized)")
    axes[2].set_title("Mean Regression RMSE"); axes[2].grid(alpha=0.3)
    axes[2].invert_yaxis()   # lower is better → goes up visually

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

# ─────────────────────────────────────────────────────────────
# Wavelength regions (re-exported for counterfactual module)
# ─────────────────────────────────────────────────────────────
WAVELENGTH_REGIONS = {
    "Blue_Green":  (0,  30),
    "Red":         (30, 60),
    "Red_edge":    (60, 80),
    "NIR":         (80, 128),
}

