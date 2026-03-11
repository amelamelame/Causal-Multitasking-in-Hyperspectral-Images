"""
models/causal/causal_model.py
===============================
FULL CAUSAL MULTITASK MODEL

Assembles all 4 causal modules around the baseline encoder:

  Input HSI patch [B, 128, 9, 9]
      ↓
  [SharedEncoder + DenseASPP]         ← same as baseline
      ↓
  [Module 2: Disentanglement]
      ├── Z_causal  [B, causal_dim, H, W]
      └── Z_spurious → Adv. Discriminator
      ↓
  [Spectral-Spatial Attention CBAM]   ← same as baseline
      ↓
  [Module 1: Causal DAG]
      └── A_soft [13×13 causal graph]
      ↓
  [Module 3: Causally-Ordered Decoders]
      ├── Classification heads (3)
      └── Regression heads (10)

  Module 4 (inference): Counterfactual explainer (post-hoc)
  Module 4b (training):  CF augmentation (called from train loop)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse baseline building blocks
from models.baseline.model import (
    SharedEncoder, CBAM, ConvBNLeaky
)
# Causal modules
from models.causal.disentangle  import CausalDisentangler, EnvironmentDiscriminator, GRL
from models.causal.dag          import CausalDAGModule
from models.causal.causal_decoder import CausalDecoderCollection
from data.envi_reader import CAT_VARIABLES, REG_VARIABLES


class CausalMultitaskModel(nn.Module):
    """
    Full causal model with all 4 modules.
    Drop-in replacement for MultitaskModel — same forward interface.
    """
    def __init__(self, cfg: dict):
        super().__init__()
        m    = cfg["model"]
        cm   = cfg.get("causal", {})
        ch   = m["encoder_channels"]
        dc   = m["decoder_channels"]
        ckpt = m["use_checkpoint"]
        nb   = cfg["data"]["num_bands"]

        # ── Shared Encoder (same as baseline) ──────────────────
        self.encoder = SharedEncoder(in_ch=nb, channels=ch,
                                     ckpt=ckpt, drop=m["dropout"])
        enc_ch = self.encoder.out_ch

        # ── Module 2: Causal Disentanglement ───────────────────
        causal_dim = cm.get("causal_dim", 64)
        spur_dim   = cm.get("spur_dim",   64)
        self.disentangle = CausalDisentangler(enc_ch, causal_dim, spur_dim)
        disent_ch  = causal_dim  # channels entering attention

        # Adapter: resize causal_dim back to enc_ch for skip connections
        self.causal_adapter = nn.Sequential(
            nn.Conv2d(causal_dim, enc_ch, 1, bias=False),
            nn.BatchNorm2d(enc_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # ── Adversarial Discriminator (Module 2) ──────────────
        n_envs = cm.get("num_envs", 7)   # TAIGA has 7 flight strips
        self.grl         = GRL(lam=cm.get("grl_lambda", 1.0))
        self.env_disc    = EnvironmentDiscriminator(causal_dim, n_envs)

        # ── Spectral-Spatial Attention Bridge ──────────────────
        self.bridge = CBAM(enc_ch, m["attn_reduction"])

        # ── Module 1: Causal DAG ───────────────────────────────
        self.dag = CausalDAGModule(
            n_vars    = 13,
            use_prior = cm.get("use_dag_prior", True),
            threshold = cm.get("dag_threshold", 0.5),
        )

        # ── Module 3: Causally-Ordered Decoders ───────────────
        skips    = {'s0':ch[0],'s1':ch[1],'s2':ch[2],'s3':ch[3]}
        cond_dim = cm.get("cond_dim", 32)
        self.decoders = CausalDecoderCollection(enc_ch, skips, dc, cond_dim)

        # Store config values used in loss computation
        self.n_envs    = n_envs
        self.causal_dim = causal_dim

    def forward(self, x: torch.Tensor,
                env_ids: torch.Tensor = None) -> dict:
        """
        x:       [B, 128, 9, 9]
        env_ids: [B] int — flight strip IDs (needed for IRM, optional at inference)

        Returns dict with keys:
          "cls":        dict[var → [B, n_cls, H, W]]
          "reg":        dict[var → [B, 1, H, W]]
          "z_causal":   [B, causal_dim, H, W]   (for IRM penalty)
          "z_spurious": [B, spur_dim, H, W]
          "env_logits": [B, n_envs]              (from discriminator)
          "x_recon":    [B, enc_ch, H, W]        (reconstruction)
          "dag_penalty":scalar                   (NOTEARS constraint)
          "dag_sparse": scalar
          "A_soft":     [13, 13]                 (soft causal adjacency)
        """
        tsz = x.shape[2:]

        # ── 1. Encode ──────────────────────────────────────────
        enc, skips = self.encoder(x)

        # ── 2. Disentangle (Module 2) ─────────────────────────
        z_c, z_s, x_recon = self.disentangle(enc)

        # Adversarial: GRL reverses gradient → Z_causal becomes env-invariant
        env_logits = self.env_disc(self.grl(z_c))   # [B, n_envs]

        # Expand Z_causal back to enc_ch for attention + decoders
        z_c_exp = self.causal_adapter(z_c)           # [B, enc_ch, H, W]

        # ── 3. Attention Bridge ────────────────────────────────
        feat = self.bridge(z_c_exp)                  # [B, enc_ch, H, W]

        # ── 4. DAG Discovery (Module 1) ───────────────────────
        A_soft, dag_pen, dag_spar = self.dag()       # A_soft: [13, 13]

        # ── 5. Causally-Ordered Decoders (Module 3) ───────────
        cls_preds, reg_preds = self.decoders(feat, skips, tsz, A_soft)

        return {
            "cls":         cls_preds,
            "reg":         reg_preds,
            "z_causal":    z_c,
            "z_spurious":  z_s,
            "env_logits":  env_logits,
            "x_recon":     x_recon,
            "dag_penalty": dag_pen,
            "dag_sparse":  dag_spar,
            "A_soft":      A_soft,
        }

    def param_count(self):
        return sum(p.numel() for p in self.parameters()) / 1e6

    def get_causal_graph(self) -> dict:
        """Return current learned causal graph (for visualization)."""
        A_soft, dag_pen, _ = self.dag()
        return {
            "A_soft":    A_soft.detach().cpu().numpy(),
            "A_hard":    (A_soft > self.dag.thresh).float().detach().cpu().numpy(),
            "dag_pen":   dag_pen.item(),
            "var_names": list(CAT_VARIABLES.keys()) + REG_VARIABLES,
        }
