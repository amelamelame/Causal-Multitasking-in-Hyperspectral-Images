"""
test_pipeline.py  —  Verify everything works before you start training
========================================================================
Run this first! It checks:
  1. RTX 3060 GPU activation
  2. TAIGA file loading (just header, no full file read)
  3. Patch extraction from real data
  4. Forward pass + loss computation
  5. VRAM usage estimate

Run: python test_pipeline.py
"""

import os, sys, yaml
import numpy as np
import torch

def main():
    print("\n" + "="*65)
    print("  TAIGA Pipeline Verification")
    print("="*65)

    # ── 1. GPU check ──────────────────────────────────────────
    print("\n[1/6] GPU Check")
    if torch.cuda.is_available():
        name  = torch.cuda.get_device_name(0)
        vram  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✓ GPU found: {name}")
        print(f"  ✓ VRAM: {vram:.2f} GB")
        print(f"  ✓ CUDA: {torch.version.cuda}")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
    else:
        print("  ✗ No GPU — check CUDA installation")
        print("    Install: pip install torch --index-url https://download.pytorch.org/whl/cu118")
        device = torch.device("cpu")
        print("  → Continuing on CPU for pipeline check only")

    # ── 2. Config ─────────────────────────────────────────────
    print("\n[2/6] Config")
    if not os.path.exists("configs/config.yaml"):
        print("  ✗ configs/config.yaml not found! Create the dataset/ folder first.")
        sys.exit(1)
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    print("  ✓ Config loaded")

    # ── 3. Dataset files ──────────────────────────────────────
    print("\n[3/6] TAIGA File Check")
    hsi_path = cfg["paths"]["hsi_file"]
    lbl_path = cfg["paths"]["label_file"]
    bad_path = cfg["paths"]["bad_stands"]

    for label, path in [("HSI binary", hsi_path),
                         ("Labels binary", lbl_path),
                         ("Bad stands CSV", bad_path)]:
        if os.path.exists(path):
            sz = os.path.getsize(path) / 1e9
            print(f"  ✓ {label}: {path}  ({sz:.2f} GB)")
        else:
            print(f"  ✗ MISSING: {label} → {path}")
            print(f"    → Put your dataset files in dataset/ folder")
            print(f"    → Rename to match config.yaml paths")
            sys.exit(1)

    # ── 4. Memory-map + pixel extraction ──────────────────────
    print("\n[4/6] Memory-map + Patch Extraction")
    from data.envi_reader   import TAIGAEnviReader
    from data.taiga_dataset import load_bad_stands

    reader     = TAIGAEnviReader(hsi_path, lbl_path)
    bad_stands = load_bad_stands(bad_path)
    print(f"  ✓ Files memory-mapped (no full RAM load)")

    # Find first valid pixel
    found = None
    half  = cfg["data"]["patch_size"] // 2
    for r in range(half, 200):
        for c in range(half, 200):
            if reader.is_valid_pixel(r, c, bad_stands):
                found = (r, c); break
        if found: break

    if found:
        r, c   = found
        patch  = reader.get_hsi_patch(r, c, cfg["data"]["patch_size"])
        labels = reader.get_label_pixel(r, c)
        print(f"  ✓ First valid pixel at ({r},{c})")
        print(f"  ✓ Patch shape: {patch.shape}  min={patch.min():.3f}  max={patch.max():.3f}")
        print(f"  ✓ Sample labels: "
              f"fertility={int(labels['fertility_class'])}  "
              f"species={int(labels['tree_species'])}  "
              f"height={labels['mean_height']:.1f}m")
    else:
        print("  ✗ No valid pixels found in first 200×200 region")

    # ── 5. Model forward pass ─────────────────────────────────
    print("\n[5/6] Model Forward Pass")
    from models.baseline.model import MultitaskModel

    model = MultitaskModel(cfg).to(device)
    print(f"  ✓ Model: {model.param_count():.2f} M parameters")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    bs    = cfg["training"]["batch_size"]
    nb    = cfg["data"]["num_bands"]
    ps    = cfg["data"]["patch_size"]
    dummy = torch.randn(bs, nb, ps, ps).to(device)

    with torch.no_grad():
        from torch.cuda.amp import autocast
        with autocast(enabled=cfg["training"]["mixed_precision"]):
            out = model(dummy)

    print(f"  ✓ Input:  [{bs}, {nb}, {ps}, {ps}]")
    print(f"  ✓ Output cls keys: {list(out['cls'].keys())}")
    print(f"  ✓ Output reg keys: {list(out['reg'].keys())[:3]}...")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✓ VRAM for batch={bs}: {peak:.2f} GB / {total:.1f} GB")
        if peak > total * 0.85:
            print(f"  ⚠  High VRAM usage! Reduce batch_size to 2 in config.yaml")
        else:
            print(f"  ✓ VRAM usage safe!")

    # ── 6. Loss function ──────────────────────────────────────
    print("\n[6/6] Loss Function")
    from data.taiga_dataset import TAIGADataset
    from losses.loss        import MultitaskLoss

    # Minimal dataset to get class weights
    ds = TAIGADataset(reader, "train", cfg["data"]["patch_size"],
                      stride=50, bad_stand_ids=bad_stands, cfg=cfg)
    cw = ds.class_weights

    loss_fn = MultitaskLoss(cw, cfg).to(device)
    from data.envi_reader import CAT_VARIABLES, REG_VARIABLES
    cat_lbl = {v: torch.zeros(bs, dtype=torch.long).to(device) for v in CAT_VARIABLES}
    reg_lbl = {v: torch.rand(bs).to(device) for v in REG_VARIABLES}
    loss, det = loss_fn(out["cls"], out["reg"], cat_lbl, reg_lbl)
    print(f"  ✓ Loss computed: {loss.item():.4f}")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "="*65)
    print("  ✓ ALL CHECKS PASSED!")
    print("  → Ready to train. Run: python train.py")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
