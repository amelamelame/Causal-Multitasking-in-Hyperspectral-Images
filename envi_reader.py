"""
data/envi_reader.py
====================
Reads the actual TAIGA ENVI binary files using memory mapping.
Handles BSQ (HSI) and BIP (labels) interleave formats.
NO full file loading — only accessed tiles go into RAM.

File specs from headers:
  HSI:    12826 x 12143 x 128 bands, int16, BSQ, scale /10000
  Labels: 12826 x 12143 x 14 bands, int32, BIP
"""

import numpy as np
import os


# ─────────────────────────────────────────────────────────────
# Scaling factors for each label band (from forestdata_stands.hdr)
# ─────────────────────────────────────────────────────────────
BAND_NAMES = [
    "fertility_class",          # Band 0  — categorical (1-6)
    "soil_class",               # Band 1  — categorical (1-2)
    "tree_species",             # Band 2  — categorical (1-3)
    "basal_area",               # Band 3  — stored *100, divide by 100
    "mean_dbh",                 # Band 4  — stored *100, divide by 100
    "stem_density",             # Band 5  — raw [1/ha]
    "mean_height",              # Band 6  — stored *100, divide by 100
    "pct_pine",                 # Band 7  — raw [%]
    "pct_spruce",               # Band 8  — raw [%]
    "pct_birch",                # Band 9  — raw [%]
    "woody_biomass",            # Band 10 — raw [t/ha]
    "leaf_area_index",          # Band 11 — stored *100, divide by 100
    "eff_leaf_area_index",      # Band 12 — stored *100, divide by 100
    "stand_id",                 # Band 13 — stand ID (not used for prediction)
]

# Divide stored values by these to get real-world values
LABEL_SCALE = {
    "basal_area":       100.0,
    "mean_dbh":         100.0,
    "mean_height":      100.0,
    "leaf_area_index":  100.0,
    "eff_leaf_area_index": 100.0,
}

# Categorical variables with their class counts (excluding 0=None)
CAT_VARIABLES = {
    "fertility_class": 6,   # classes 1-6
    "soil_class":      2,   # classes 1-2
    "tree_species":    3,   # classes 1-3
}

# Regression variables (continuous)
REG_VARIABLES = [
    "basal_area", "mean_dbh", "stem_density", "mean_height",
    "pct_pine", "pct_spruce", "pct_birch",
    "woody_biomass", "leaf_area_index", "eff_leaf_area_index",
]

# Human-readable class names for visualization
CLASS_NAMES = {
    "fertility_class": {1:"Herb-rich", 2:"Herb-rich heath",
                        3:"Mesic heath", 4:"Sub-xeric",
                        5:"Xeric heath", 6:"Barren heath"},
    "soil_class":      {1:"Mineral", 2:"Organic peat"},
    "tree_species":    {1:"Scots pine", 2:"Norway spruce", 3:"Birch"},
}


class TAIGAEnviReader:
    """
    Memory-mapped reader for TAIGA ENVI binary files.
    Access any pixel or tile without loading the full 37 GB file.
    """

    def __init__(self, hsi_path: str, label_path: str):
        """
        hsi_path:   path to '20170615_reflectance_mosaic_128b' (no .hdr)
        label_path: path to 'forestdata_stands' (no .hdr)
        """
        # ── Image dimensions ──
        self.lines   = 12143
        self.samples = 12826
        self.hsi_bands   = 128
        self.label_bands = 14

        # ── Memory-map HSI (BSQ, int16) ──
        # BSQ layout: [bands, lines, samples]
        hsi_bytes = self.hsi_bands * self.lines * self.samples * 2  # int16 = 2 bytes
        self._hsi_raw = np.memmap(
            hsi_path, dtype=np.int16, mode='r',
            shape=(self.hsi_bands, self.lines, self.samples)
        )
        print(f"[ENVI] HSI   mapped: {self.hsi_bands}b × {self.lines}L × {self.samples}S  "
              f"({hsi_bytes/1e9:.1f} GB)")

        # ── Memory-map Labels (BIP, int32) ──
        # BIP layout: [lines, samples, bands]
        lbl_bytes = self.lines * self.samples * self.label_bands * 4  # int32 = 4 bytes
        self._lbl_raw = np.memmap(
            label_path, dtype=np.int32, mode='r',
            shape=(self.lines, self.samples, self.label_bands)
        )
        print(f"[ENVI] Labels mapped: {self.lines}L × {self.samples}S × {self.label_bands}b  "
              f"({lbl_bytes/1e9:.1f} GB)")

    def get_hsi_patch(self, row: int, col: int, patch: int) -> np.ndarray:
        """
        Extract HSI patch centered at (row, col).
        Returns: [128, patch, patch] float32, normalized to [0, 1]
        """
        h = patch // 2
        r0, r1 = row - h, row + h + 1
        c0, c1 = col - h, col + h + 1
        patch_data = self._hsi_raw[:, r0:r1, c0:c1].astype(np.float32)
        # Scale by /10000 (reflectance scale factor from header)
        patch_data /= 10000.0
        patch_data = np.clip(patch_data, 0, 1)
        return patch_data   # [128, patch_h, patch_w]

    def get_label_pixel(self, row: int, col: int) -> dict:
        """
        Get all 13 labels (excluding standID) for center pixel.
        Returns dict with real-world values.
        """
        raw = self._lbl_raw[row, col, :]   # [14]
        result = {}
        for i, name in enumerate(BAND_NAMES[:-1]):   # skip stand_id
            val = int(raw[i])
            if name in LABEL_SCALE:
                result[name] = float(val) / LABEL_SCALE[name]
            else:
                result[name] = float(val)
        result["stand_id"] = int(raw[13])
        return result

    def get_hsi_tile(self, row_start: int, row_end: int,
                     col_start: int, col_end: int) -> np.ndarray:
        """
        Extract a rectangular tile from the HSI for prediction maps.
        Returns: [128, H, W] float32
        """
        tile = self._hsi_raw[:, row_start:row_end, col_start:col_end].astype(np.float32)
        tile /= 10000.0
        tile = np.clip(tile, 0, 1)
        return tile

    def get_label_tile(self, row_start: int, row_end: int,
                       col_start: int, col_end: int) -> np.ndarray:
        """
        Extract label tile.
        Returns: [H, W, 14] int32
        """
        return self._lbl_raw[row_start:row_end, col_start:col_end, :].copy()

    def is_valid_pixel(self, row: int, col: int,
                       bad_stand_ids: set = None) -> bool:
        """Check if pixel has valid labels (not 0=None for all cat vars)."""
        raw = self._lbl_raw[row, col, :]
        # All cat variables must be non-zero
        if raw[0] == 0 or raw[1] == 0 or raw[2] == 0:
            return False
        # Exclude bad stands
        if bad_stand_ids and int(raw[13]) in bad_stand_ids:
            return False
        # HSI must not be all-zero (missing data)
        if self._hsi_raw[64, row, col] == 0:
            return False
        return True

    def get_rgb(self, row_start: int, row_end: int,
                col_start: int, col_end: int) -> np.ndarray:
        """
        Pseudo-RGB image using default bands from header: 97, 65, 33.
        Returns: [H, W, 3] uint8
        """
        r = self._hsi_raw[96, row_start:row_end, col_start:col_end].astype(np.float32)
        g = self._hsi_raw[64, row_start:row_end, col_start:col_end].astype(np.float32)
        b = self._hsi_raw[32, row_start:row_end, col_start:col_end].astype(np.float32)

        rgb = np.stack([r, g, b], axis=-1)
        # Percentile stretch for visualization
        for c in range(3):
            p2  = np.percentile(rgb[:,:,c][rgb[:,:,c]>0], 2)
            p98 = np.percentile(rgb[:,:,c][rgb[:,:,c]>0], 98)
            rgb[:,:,c] = np.clip((rgb[:,:,c] - p2) / (p98 - p2 + 1e-8), 0, 1)

        return (rgb * 255).astype(np.uint8)
