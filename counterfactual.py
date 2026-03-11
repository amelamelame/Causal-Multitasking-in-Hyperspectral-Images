"""
models/causal/counterfactual.py
================================
MODULE 4 — Counterfactual Explanations + Data Augmentation

4a: Counterfactual Explainer (Wachter et al. 2018)
    For any input pixel, finds the MINIMAL spectral change needed
    to flip the prediction to a target class.
    Maps the delta to wavelength regions for physical interpretation.

4b: Counterfactual Data Augmentation
    Generates synthetic minority-class samples by perturbing majority
    class pixels toward the minority class using model gradients.
    More physically valid than SMOTE (which interpolates blindly).

TAIGA wavelength bands (from header, 128 bands):
  Bands 1–30   : Blue/Green   (400–530 nm)
  Bands 31–60  : Red          (530–660 nm)
  Bands 61–80  : Red-edge     (660–750 nm)   ← key for vegetation
  Bands 81–128 : NIR          (750–991 nm)   ← key for LAI, biomass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict


# Wavelength regions for TAIGA AISA Eagle II (128 bands, 400–991 nm)
WAVELENGTH_REGIONS = {
    "Blue_Green":  (0,  30),   # 400–530 nm
    "Red":         (30, 60),   # 530–660 nm
    "Red_edge":    (60, 80),   # 660–750 nm
    "NIR":         (80, 128),  # 750–991 nm
}


# ─────────────────────────────────────────────────────────────
# Module 4a: Counterfactual Explainer
# ─────────────────────────────────────────────────────────────

class CounterfactualExplainer:
    """
    Finds minimal spectral perturbation to change a model's prediction.

    For a pixel with predicted class C, finds Δx such that:
      argmax(model(x + Δx)) = target_class
      ||Δx||₁ is minimized  (minimal spectral change)

    Optimization objective:
      min  λ_dist * ||Δx||₁  +  CE(model(x+Δx), target_class)
    """
    def __init__(self, model: nn.Module, device: torch.device,
                 max_iter: int = 200, lr: float = 0.01,
                 lambda_dist: float = 0.5):
        self.model      = model
        self.device     = device
        self.max_iter   = max_iter
        self.lr         = lr
        self.lam        = lambda_dist
        self.model.eval()

    @torch.no_grad()
    def explain(
        self,
        patch: torch.Tensor,        # [1, 128, 9, 9] — single pixel patch
        task:  str,                 # e.g. "tree_species"
        target_class: int,          # desired class to flip to
        is_cls: bool = True,
    ) -> Dict:
        """
        Returns:
            delta:         [128] spectral perturbation
            cf_patch:      [1, 128, 9, 9] counterfactual patch
            success:       bool — did prediction flip?
            region_import: dict of wavelength region importance
            n_iter:        iterations until convergence
        """
        patch = patch.to(self.device).float()

        # Δx is the perturbation — we optimize this
        delta = torch.zeros(1, 128, 1, 1, device=self.device, requires_grad=True)
        opt   = torch.optim.Adam([delta], lr=self.lr)

        target_t = torch.tensor([target_class], device=self.device)
        success  = False

        for i in range(self.max_iter):
            opt.zero_grad()
            # Apply spectral perturbation (same delta to all spatial positions)
            cf_patch = patch + delta.expand_as(patch)
            cf_patch = torch.clamp(cf_patch, 0, 1)

            with torch.enable_grad():
                out = self.model(cf_patch)

            if is_cls:
                pred_logits = out["cls"][task]            # [1, n_cls, H, W]
                H, W = pred_logits.shape[2:]
                pred_ctr = pred_logits[0, :, H//2, W//2] # [n_cls]
                # CE loss: push toward target class
                task_loss = F.cross_entropy(pred_ctr.unsqueeze(0), target_t)
                pred_cls  = pred_ctr.argmax().item()
            else:
                pred_val  = out["reg"][task][0, 0, patch.shape[2]//2, patch.shape[3]//2]
                task_loss = (pred_val - target_class) ** 2
                pred_cls  = -1

            # Distance: L1 sparsity on spectral perturbation
            dist_loss = delta.abs().mean()

            loss = task_loss + self.lam * dist_loss
            loss.backward()
            opt.step()

            # Check if prediction flipped
            if is_cls and pred_cls == target_class:
                success = True
                break

        # ── Compute wavelength region importance ──
        delta_np = delta.detach().cpu().numpy().squeeze()   # [128]
        delta_abs = np.abs(delta_np)

        region_import = {}
        for name, (b0, b1) in WAVELENGTH_REGIONS.items():
            region_import[name] = float(delta_abs[b0:b1].mean())

        # Normalize to sum = 1
        total = sum(region_import.values()) + 1e-8
        region_import = {k: v/total for k, v in region_import.items()}

        cf_patch_out = (patch + delta.expand_as(patch)).detach().clamp(0, 1)

        return {
            "delta":          delta_np,
            "cf_patch":       cf_patch_out,
            "success":        success,
            "region_import":  region_import,
            "n_iter":         i + 1,
        }

    def batch_explain(self, patches, task, target_classes, is_cls=True):
        """Run explain() for a batch and aggregate region importances."""
        results = []
        for i, (patch, tc) in enumerate(zip(patches, target_classes)):
            r = self.explain(patch.unsqueeze(0), task, int(tc), is_cls)
            results.append(r)

        # Aggregate: mean region importance across samples
        mean_import = {}
        for key in WAVELENGTH_REGIONS:
            mean_import[key] = np.mean([r["region_import"][key] for r in results])
        success_rate = np.mean([r["success"] for r in results])

        return results, mean_import, success_rate


# ─────────────────────────────────────────────────────────────
# Module 4b: Counterfactual Data Augmentation
# ─────────────────────────────────────────────────────────────

class CounterfactualAugmentor:
    """
    Generates synthetic minority-class samples using model gradients.

    Algorithm:
      1. Find majority-class pixels with confident predictions
      2. Compute gradient ∇_x CE(model(x), minority_class)
      3. Take small step in gradient direction: x_aug = x + α * ∇_x
      4. This creates a spectrally-similar pixel that the model thinks
         is closer to the minority class

    This is more physically meaningful than SMOTE because:
    - It uses the model's own learned decision boundary
    - Steps respect spectral correlations across bands
    - Each generated sample is causally consistent
    """
    def __init__(self, model: nn.Module, device: torch.device,
                 step_size: float = 0.02, n_steps: int = 5):
        self.model     = model
        self.device    = device
        self.step_size = step_size
        self.n_steps   = n_steps

    def augment(
        self,
        patches:    torch.Tensor,   # [N, 128, 9, 9] majority class patches
        task:       str,            # categorical task name
        target_cls: int,            # minority class to generate toward
        n_samples:  int = 50,
    ) -> torch.Tensor:
        """
        Returns:
            aug_patches: [n_samples, 128, 9, 9] — synthetic minority-class patches
        """
        self.model.eval()
        aug_list = []
        target_t = torch.tensor([target_cls], device=self.device)

        indices = torch.randperm(len(patches))[:n_samples]

        for idx in indices:
            patch = patches[idx:idx+1].clone().to(self.device).float()
            patch.requires_grad_(True)

            for _ in range(self.n_steps):
                out   = self.model(patch)
                logit = out["cls"][task]
                H, W  = logit.shape[2:]
                pred  = logit[0, :, H//2, W//2].unsqueeze(0)
                loss  = F.cross_entropy(pred, target_t)

                grad = torch.autograd.grad(loss, patch, retain_graph=False)[0]
                # Spectral perturbation: step in negative gradient direction
                # (minimizing loss = moving toward target class)
                with torch.no_grad():
                    patch = patch - self.step_size * grad.sign()
                    patch = torch.clamp(patch, 0, 1)
                patch.requires_grad_(True)

            aug_list.append(patch.detach().cpu())

        return torch.cat(aug_list, dim=0) if aug_list else patches[:n_samples]

    def get_augmented_batch(
        self,
        batch: dict,
        task:  str,
        target_cls: int,
    ) -> dict:
        """
        Takes a training batch, generates CF augmented samples for minority class,
        and returns an augmented batch ready for the training loop.
        """
        patches = batch["patch"]
        aug_p   = self.augment(patches, task, target_cls,
                               n_samples=min(len(patches)//2, 16))

        # Create labels for augmented samples
        aug_cat = {v: (torch.full((len(aug_p),), target_cls, dtype=torch.long)
                       if v == task else
                       batch["cat"][v][:len(aug_p)])
                   for v in batch["cat"]}
        aug_reg = {v: batch["reg"][v][:len(aug_p)] for v in batch["reg"]}

        return {
            "patch":  torch.cat([patches, aug_p.to(patches.device)]),
            "cat":    {v: torch.cat([batch["cat"][v], aug_cat[v].to(patches.device)])
                       for v in aug_cat},
            "reg":    {v: torch.cat([batch["reg"][v], aug_reg[v].to(patches.device)])
                       for v in aug_reg},
        }
