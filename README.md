# Multitask Deep Learning for TAIGA Dataset
## Complete Guide — RTX 3060 6GB / 16GB RAM

---

## STEP 0 — Project Folder Layout

After unzipping, your folder must look like this:

```
taiga_project/
├── dataset/                         ← PUT YOUR TAIGA FILES HERE
│   ├── 20170615_reflectance_mosaic_128b      (3.89 GB binary)
│   ├── 20170615_reflectance_mosaic_128b.hdr  (7 KB header)
│   ├── forestdata_stands                      (851 MB binary)
│   ├── forestdata_stands.hdr                  (3 KB header)
│   └── bad_stands_updated_15092020.csv        (2 KB)
│
├── configs/
│   └── config.yaml
├── data/
│   ├── envi_reader.py
│   └── taiga_dataset.py
├── models/baseline/
│   └── model.py
├── losses/
│   └── loss.py
├── evaluation/
│   └── metrics.py
├── visualization/
│   └── prediction_maps.py
├── utils/
│   └── gpu_utils.py
├── train.py
├── evaluate.py
├── test_pipeline.py
└── requirements.txt
```

---

## STEP 1 — Install Python Environment

Open terminal/CMD in the project folder:

```bash
# Create conda environment
conda create -n taiga python=3.9 -y
conda activate taiga

# Install PyTorch with CUDA (check your CUDA version first with: nvidia-smi)
# For CUDA 11.8 (most common on RTX 3060 laptops):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other packages
pip install -r requirements.txt
```

---

## STEP 2 — Place Dataset Files

Copy your TAIGA files into the `dataset/` folder:

```
20170615_reflectance_mosaic_128b      → dataset/20170615_reflectance_mosaic_128b
20170615_reflectance_mosaic_128b.hdr  → dataset/20170615_reflectance_mosaic_128b.hdr
forestdata_stands                      → dataset/forestdata_stands
forestdata_stands.hdr                  → dataset/forestdata_stands.hdr
bad_stands_updated_15092020.csv        → dataset/bad_stands_updated_15092020.csv
```

The two large binary files (no extension) are the actual image data.
**Do NOT rename them** — the config already uses the correct names.

---

## STEP 3 — Verify GPU + Pipeline

```bash
python test_pipeline.py
```

Expected output:
```
[1/6] GPU Check
  ✓ GPU found: NVIDIA GeForce RTX 3060 Laptop GPU
  ✓ VRAM: 6.14 GB
  ✓ CUDA: 11.8
[2/6] Config
  ✓ Config loaded
[3/6] TAIGA File Check
  ✓ HSI binary: dataset/20170615_reflectance_mosaic_128b  (3.89 GB)
  ✓ Labels binary: dataset/forestdata_stands  (0.85 GB)
  ✓ Bad stands CSV: dataset/bad_stands_updated_15092020.csv  (0.00 GB)
[4/6] Memory-map + Patch Extraction
  ✓ Files memory-mapped (no full RAM load)
  ✓ Patch shape: (128, 9, 9)
[5/6] Model Forward Pass
  ✓ Model: ~8.2 M parameters
  ✓ VRAM for batch=4: ~1.8 GB / 6.1 GB  ✓ safe!
[6/6] Loss Function
  ✓ Loss computed
  ✓ ALL CHECKS PASSED!
```

If VRAM shows >5 GB → open `configs/config.yaml` and change `batch_size: 2`

---

## STEP 4 — Start Training

```bash
python train.py
```

Monitor progress in a second terminal:
```bash
tensorboard --logdir results/logs
# Then open browser: http://localhost:6006
```

**Training time estimates on RTX 3060:**

| Config | Time/epoch | 150 epochs |
|--------|-----------|------------|
| batch=4, patch=9  | ~20-40 min | 2–4 days |
| batch=2, patch=9  | ~10-20 min | 1–2 days |

> **TIP:** Run overnight: `python train.py > results/log.txt 2>&1 &`
> Check progress: `tail -f results/log.txt`

Training auto-saves:
- `checkpoints/best.pth`   ← best validation RMSE (use this for results)
- `checkpoints/latest.pth` ← most recent (use for resuming)

---

## STEP 5 — Resume if Interrupted

```bash
python train.py --resume checkpoints/latest.pth
```

---

## STEP 6 — Generate Evaluation + Maps

After training finishes (or even mid-training using best.pth):

```bash
python evaluate.py --checkpoint checkpoints/best.pth
```

This generates in `results/maps/`:

| File | What it shows |
|------|--------------|
| `summary_dashboard.png`   | **Best for presentation** — all results on one slide |
| `classification_maps.png` | RGB + GT map + Predicted map + Error map for 3 class tasks |
| `regression_maps.png`     | GT / Predicted / Error for all 10 continuous variables |
| `training_curves.png`     | Loss + OA + RMSE over training epochs |

---

## STEP 7 — Read Your Results

Open `results/eval_metrics.json` for numeric results. Key metrics:

```
Classification (per task):
  OA   = Overall Accuracy (%)         → higher is better
  MCA  = Mean Class Accuracy (%)      → better for imbalanced data
  F1   = F1-score (%)
  AUC  = ROC-AUC (%)

Regression (per task):
  RMSE = Root Mean Square Error       → lower is better
  MAE  = Mean Absolute Error          → lower is better
  R²   = Coefficient of Determination → 1.0 is perfect
```

---

## IF YOU GET ERRORS

### "CUDA out of memory"
```yaml
# In configs/config.yaml, change:
training:
  batch_size: 2        # reduce from 4 to 2
model:
  encoder_channels: [32, 64, 64, 128]   # halve channels
```

### "File not found"
Check that your dataset files are in `dataset/` and named exactly as in config.yaml.
The binary files have NO extension — just the name without .hdr.

### "No valid pixels found"
The bad_stands CSV excludes 167 degraded stands. This is normal — those pixels
are filtered out. There are still millions of valid pixels.

### Training is very slow (no GPU)
```bash
# Confirm GPU is being used:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Should print: True  NVIDIA GeForce RTX 3060 ...
```

---

## Moving to Google Colab Pro (Phase 2)

1. Zip the project: your laptop files + trained checkpoint
2. Upload to Google Drive
3. Mount Drive in Colab: `from google.colab import drive; drive.mount('/content/drive')`
4. Update config.yaml for Colab:
   ```yaml
   batch_size: 32
   encoder_channels: [128, 256, 256, 512]
   patch_size: 11
   num_workers: 4
   pin_memory: true
   seeds: [42, 0, 1, 2, 3, 7, 13, 21, 99, 123]
   ```
5. Run 10 seeds for paper-quality results

---

## What Each Output File Means for Your Presentation

**`summary_dashboard.png`** — Show this first to your professor:
- Top row: RGB image + 3 classification predictions with accuracy
- Middle rows: 8 key regression predictions with error maps
- Right panel: complete metrics table

**`classification_maps.png`** — Show this for classification detail:
- Column 1: Ground truth (what the labels say)
- Column 2: Model prediction (what your model says) — with legend
- Column 3: Green = correct pixel, Red = wrong pixel

**`regression_maps.png`** — Show this for regression detail:
- Color intensity = value magnitude (e.g. dark green = dense forest)
- Error map shows where model struggles most (typically forest edges)

**`training_curves.png`** — Shows model learned progressively:
- Loss decreasing = model is learning
- OA increasing + RMSE decreasing = model getting better
