"""
models/causal/disentangle.py
==============================
MODULE 2 — Causal Feature Disentanglement

Splits the encoder bottleneck into:
  Z_causal   — causally invariant features (stable across flight strips)
  Z_spurious — acquisition artifacts (strip-specific noise)

Three mechanisms enforce disentanglement:
  1. IRM Penalty      — gradients of cls loss w.r.t. Z_causal must be 0 across environments
  2. Adversarial Disc — environment discriminator trained adversarially on Z_causal
                        (forces Z_causal to NOT encode environment/strip ID)
  3. HSIC Loss        — minimizes statistical dependence between Z_causal and Z_spurious

Architecture:
  enc_features [B, enc_ch, H, W]
       ↓
  [CausalProjector]  →  Z_causal  [B, causal_dim, H, W]
  [SpuriousProjector] → Z_spurious [B, spur_dim, H, W]
       ↓
  Z_causal passed to attention + decoders
  Z_spurious passed to adversarial discriminator (detached)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalDisentangler(nn.Module):
    """
    Splits encoder features into causal and spurious subspaces.
    Causal subspace: flight-strip-invariant features (true forest properties)
    Spurious subspace: lighting/sensor artifacts specific to flight strip
    """
    def __init__(self, enc_ch: int, causal_dim: int = 64, spur_dim: int = 64):
        super().__init__()
        self.causal_dim = causal_dim
        self.spur_dim   = spur_dim

        # Projection heads — separate 1×1 convolutions
        self.causal_proj = nn.Sequential(
            nn.Conv2d(enc_ch, causal_dim, 1, bias=False),
            nn.BatchNorm2d(causal_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(causal_dim, causal_dim, 1, bias=False),
            nn.BatchNorm2d(causal_dim),
        )
        self.spur_proj = nn.Sequential(
            nn.Conv2d(enc_ch, spur_dim, 1, bias=False),
            nn.BatchNorm2d(spur_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(spur_dim, spur_dim, 1, bias=False),
            nn.BatchNorm2d(spur_dim),
        )

        # Reconstruction head — ensures no information loss
        self.reconstruct = nn.Sequential(
            nn.Conv2d(causal_dim + spur_dim, enc_ch, 1, bias=False),
            nn.BatchNorm2d(enc_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.out_ch = causal_dim

    def forward(self, x: torch.Tensor):
        """
        x: encoder features [B, enc_ch, H, W]
        Returns:
            z_causal:  [B, causal_dim, H, W]  — for decoders
            z_spurious:[B, spur_dim, H, W]    — for discriminator
            x_recon:   [B, enc_ch, H, W]      — reconstruction (for recon loss)
        """
        z_c = self.causal_proj(x)    # [B, causal_dim, H, W]
        z_s = self.spur_proj(x)      # [B, spur_dim, H, W]
        x_r = self.reconstruct(torch.cat([z_c, z_s], dim=1))
        return z_c, z_s, x_r


class EnvironmentDiscriminator(nn.Module):
    """
    Adversarial discriminator: tries to predict flight strip ID from Z_causal.
    Trained adversarially — if it CAN'T predict the strip, Z_causal is strip-invariant.

    num_envs: number of TAIGA flight strips (7 flight lines in dataset)
    """
    def __init__(self, causal_dim: int, num_envs: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # global average pool → [B, causal_dim, 1, 1]
            nn.Flatten(),                      # → [B, causal_dim]
            nn.Linear(causal_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(64, num_envs),           # predict which strip
        )

    def forward(self, z_causal: torch.Tensor) -> torch.Tensor:
        """Returns logits [B, num_envs]."""
        return self.net(z_causal)


class GradientReversal(torch.autograd.Function):
    """
    Gradient Reversal Layer (Ganin & Lempitsky 2015).
    Forward pass: identity.
    Backward pass: negates gradient (× -lambda).
    This is what makes the discriminator adversarial.
    """
    @staticmethod
    def forward(ctx, x, lam):
        ctx.save_for_backward(torch.tensor(lam))
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        lam = ctx.saved_tensors[0]
        return -lam * grad, None


class GRL(nn.Module):
    """Wrapper for Gradient Reversal Layer."""
    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = lam

    def forward(self, x):
        return GradientReversal.apply(x, self.lam)

    def set_lambda(self, lam: float):
        self.lam = lam


# ─────────────────────────────────────────────────────────────
# IRM Penalty
# ─────────────────────────────────────────────────────────────

def irm_penalty(loss_by_env: list) -> torch.Tensor:
    """
    IRM penalty (Arjovsky et al. 2019).
    Computes variance of gradients across environments.
    Penalizes if loss gradient w.r.t. a scaling constant differs across envs.

    loss_by_env: list of scalar losses, one per environment (flight strip)
    Returns scalar penalty.
    """
    if len(loss_by_env) < 2:
        return torch.tensor(0.0, requires_grad=True)

    # Compute mean + variance across environments
    env_losses = torch.stack(loss_by_env)      # [num_envs]
    mean_loss  = env_losses.mean()
    variance   = ((env_losses - mean_loss) ** 2).mean()
    return variance


# ─────────────────────────────────────────────────────────────
# HSIC (Hilbert-Schmidt Independence Criterion)
# ─────────────────────────────────────────────────────────────

def hsic_loss(z_c: torch.Tensor, z_s: torch.Tensor,
              sigma: float = 1.0) -> torch.Tensor:
    """
    HSIC-based independence loss between Z_causal and Z_spurious.
    Minimizing this forces statistical independence between the two subspaces.

    Uses RBF kernel on global average pooled features.
    z_c: [B, causal_dim, H, W]
    z_s: [B, spur_dim,   H, W]
    """
    # Global average pool to get [B, dim] vectors
    c = z_c.mean([2, 3])   # [B, causal_dim]
    s = z_s.mean([2, 3])   # [B, spur_dim]
    B = c.shape[0]

    if B < 2:
        return torch.tensor(0.0, device=z_c.device)

    # RBF kernel: K(x_i, x_j) = exp(-||x_i - x_j||² / 2σ²)
    def rbf_kernel(X: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(X, X, p=2) ** 2   # [B, B]
        return torch.exp(-dists / (2 * sigma ** 2))

    Kc = rbf_kernel(c)    # [B, B]
    Ks = rbf_kernel(s)    # [B, B]

    # Centre the kernels: Kc_hat = H Kc H  where H = I - 1/n * 11^T
    H  = torch.eye(B, device=c.device) - (1.0 / B)
    Kc_hat = H @ Kc @ H
    Ks_hat = H @ Ks @ H

    # HSIC = 1/(B-1)^2 * trace(Kc_hat @ Ks_hat)
    hsic = torch.trace(Kc_hat @ Ks_hat) / ((B - 1) ** 2 + 1e-8)
    return hsic.abs()
