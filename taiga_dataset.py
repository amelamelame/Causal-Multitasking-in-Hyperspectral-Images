"""
data/taiga_dataset.py
======================
PyTorch Dataset using the real TAIGA ENVI files.
Spatial train/test split — no pixel overlap between sets.
"""

import os, csv, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple
from .envi_reader import (
    TAIGAEnviReader, CAT_VARIABLES, REG_VARIABLES,
    BAND_NAMES, LABEL_SCALE
)


def load_bad_stands(csv_path: str) -> set:
    bad = set()
    if not os.path.exists(csv_path):
        return bad
    with open(csv_path) as f:
        for row in csv.reader(f):
            if row and row[0].isdigit():
                bad.add(int(row[0]))
    print(f"[Dataset] Loaded {len(bad)} bad stand IDs to exclude.")
    return bad


class TAIGADataset(Dataset):
    """
    Extracts patches from the real TAIGA ENVI files on-the-fly.
    Memory-mapped — only accessed patches go into RAM.
    """

    def __init__(
        self,
        reader:        TAIGAEnviReader,
        split:         str,           # "train" | "val" | "test"
        patch_size:    int   = 9,
        stride:        int   = 8,
        bad_stand_ids: set   = None,
        augment:       bool  = True,
        cfg:           dict  = None,
    ):
        self.reader    = reader
        self.split     = split
        self.patch_size = patch_size
        self.half       = patch_size // 2
        self.stride     = stride
        self.bad_stands = bad_stand_ids or set()
        self.augment    = augment and (split == "train")

        # ── Spatial split columns ──
        c = cfg["data"] if cfg else {}
        train_end   = c.get("train_col_end",  8997)
        test_start  = c.get("test_col_start", 9000)

        if split == "train":
            self.col_range = (self.half, train_end - self.half)
        elif split == "val":
            # Middle 10% of train area for validation
            mid = train_end // 2
            self.col_range = (mid - 500, mid + 500)
        else:  # test
            self.col_range = (test_start + self.half,
                              reader.samples - self.half)

        self.row_range = (self.half, reader.lines - self.half)

        print(f"\n[Dataset] Building {split} index...")
        t0 = time.time()
        self._build_index()
        print(f"[Dataset] {split}: {len(self.indices):,} valid patches  "
              f"({time.time()-t0:.1f}s)")

        # Compute class weights and normalization stats
        self._compute_class_weights()
        self._compute_reg_stats()

    def _build_index(self):
        """Pre-compute valid (row, col) pixel indices."""
        reader = self.reader
        indices = []

        row_range = range(self.row_range[0], self.row_range[1], self.stride)
        col_range = range(self.col_range[0], self.col_range[1], self.stride)

        for row in row_range:
            for col in col_range:
                if reader.is_valid_pixel(row, col, self.bad_stands):
                    indices.append((row, col))

        self.indices = indices

    def _compute_class_weights(self):
        """Inverse median frequency for each categorical task."""
        print(f"  Computing class weights for {self.split}...")
        self.class_weights = {}
        counts_accum = {v: np.zeros(n+1) for v, n in CAT_VARIABLES.items()}

        # Sample subset for speed (max 50k pixels)
        sample_idx = self.indices[::max(1, len(self.indices)//50000)]
        for row, col in sample_idx:
            lbl = self.reader._lbl_raw[row, col, :]
            for i, var in enumerate(["fertility_class","soil_class","tree_species"]):
                v = int(lbl[i])
                if 0 <= v < len(counts_accum[var]):
                    counts_accum[var][v] += 1

        for var, counts in counts_accum.items():
            counts[0] = 0  # ignore class 0 (None)
            total = counts.sum()
            freqs = np.where(counts > 0, counts / (total + 1e-8), 0)
            med   = np.median(freqs[freqs > 0]) if (freqs > 0).any() else 1.0
            # Inverse median frequency (paper Eq. 14)
            weights = np.where(freqs > 0, med / (freqs + 1e-8), 0.0)
            # Remap to 0-indexed (class 1 → index 0, etc.)
            n = CAT_VARIABLES[var]
            self.class_weights[var] = torch.FloatTensor(weights[1:n+1])

    def _compute_reg_stats(self):
        """Min/max for normalization of regression variables."""
        # Use known ranges from paper / README
        self.reg_min = {
            "basal_area": 0.0,     "mean_dbh": 0.0,
            "stem_density": 0.0,   "mean_height": 0.0,
            "pct_pine": 0.0,       "pct_spruce": 0.0,
            "pct_birch": 0.0,      "woody_biomass": 0.0,
            "leaf_area_index": 0.0,"eff_leaf_area_index": 0.0,
        }
        self.reg_max = {
            "basal_area": 35.51,    "mean_dbh": 30.89,
            "stem_density": 6240.0, "mean_height": 24.16,
            "pct_pine": 100.0,      "pct_spruce": 84.0,
            "pct_birch": 58.0,      "woody_biomass": 180.0,
            "leaf_area_index": 9.66,"eff_leaf_area_index": 6.45,
        }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        row, col = self.indices[idx]
        h = self.half

        # ── HSI patch [128, pH, pW] ──
        patch = self.reader.get_hsi_patch(row, col, self.patch_size)

        # ── Augmentation ──
        if self.augment:
            if np.random.random() > 0.5:
                patch = np.flip(patch, axis=2).copy()
            if np.random.random() > 0.5:
                patch = np.flip(patch, axis=1).copy()

        # ── Labels ──
        raw = self.reader._lbl_raw[row, col, :]

        # Categorical: remap to 0-indexed (stored as 1-6, model needs 0-5)
        cat = {
            "fertility_class": torch.tensor(max(0, int(raw[0]) - 1), dtype=torch.long),
            "soil_class":      torch.tensor(max(0, int(raw[1]) - 1), dtype=torch.long),
            "tree_species":    torch.tensor(max(0, int(raw[2]) - 1), dtype=torch.long),
        }

        # Regression: apply scale + min-max normalize to [0, 1]
        band_map = {
            "basal_area":3, "mean_dbh":4, "stem_density":5, "mean_height":6,
            "pct_pine":7, "pct_spruce":8, "pct_birch":9,
            "woody_biomass":10, "leaf_area_index":11, "eff_leaf_area_index":12,
        }
        reg = {}
        for var in REG_VARIABLES:
            bi  = band_map[var]
            val = float(raw[bi])
            # Apply TAIGA scaling
            if var in LABEL_SCALE:
                val /= LABEL_SCALE[var]
            # Min-max normalize
            vmin = self.reg_min[var]
            vmax = self.reg_max[var]
            norm = np.clip((val - vmin) / (vmax - vmin + 1e-8), 0, 1)
            reg[var] = torch.tensor(norm, dtype=torch.float32)

        # Flight strip env_id from column position (7 TAIGA flight strips)
        # Columns roughly split into 7 equal strips across 12826 width
        env_id = min(int(col / (12826 / 7)), 6)

        return {
            "patch":  torch.from_numpy(patch),   # [128, 9, 9]
            "cat":    cat,
            "reg":    reg,
            "coords": torch.tensor([row, col]),
            "env_id": torch.tensor(env_id, dtype=torch.long),
        }


def get_dataloaders(reader, bad_stands, cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    d   = cfg["data"]
    t   = cfg["training"]
    kw  = dict(reader=reader, patch_size=d["patch_size"],
                bad_stand_ids=bad_stands, cfg=cfg)

    train_ds = TAIGADataset(**kw, split="train", stride=d["train_stride"],  augment=True)
    val_ds   = TAIGADataset(**kw, split="val",   stride=d["val_stride"],    augment=False)
    test_ds  = TAIGADataset(**kw, split="test",  stride=d["val_stride"],    augment=False)

    lkw = dict(batch_size=t["batch_size"],
               num_workers=d["num_workers"],
               pin_memory=d["pin_memory"])

    return (DataLoader(train_ds, shuffle=True,  drop_last=True,  **lkw),
            DataLoader(val_ds,   shuffle=False, drop_last=False, **lkw),
            DataLoader(test_ds,  shuffle=False, drop_last=False, **lkw))
