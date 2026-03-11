"""
models/causal/causal_decoder.py
================================
MODULE 3 — Causally-Ordered Task Decoders

Each task decoder can receive conditioning signals from its causal PARENT
decoders (identified by Module 1's DAG). This lets the model exploit
causal structure during prediction.

Example:
  tree_species decoder runs first (no parents in its subgraph)
  pct_pine decoder runs next and receives tree_species features as input
  woody_biomass decoder receives mean_height + basal_area features

Architecture per task:
  shared_features + causal_parent_features
       ↓
  [DecBlocks (same as baseline)]
       ↓
  [Causal Conditioning Layer] ← weighted sum of parent task features
       ↓
  prediction head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from data.envi_reader import CAT_VARIABLES, REG_VARIABLES


# Ordered list of all 13 variables (must match dag.py ALL_VARIABLES)
ALL_VARS = (
    list(CAT_VARIABLES.keys()) + REG_VARIABLES
)


class ConvBNLeaky(nn.Module):
    def __init__(self, ic, oc, k=3, p=1, d=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic, oc, k, padding=p*d, dilation=d, bias=False),
            nn.BatchNorm2d(oc), nn.LeakyReLU(0.1, inplace=True))
    def forward(self, x): return self.net(x)


class DecBlock(nn.Module):
    def __init__(self, ic, sc, oc):
        super().__init__()
        self.red = ConvBNLeaky(ic+sc, oc, 1, p=0)
        self.res = nn.Sequential(ConvBNLeaky(oc,oc), ConvBNLeaky(oc,oc))
        self.skip = nn.Sequential(nn.Conv2d(oc,oc,1,bias=False),nn.BatchNorm2d(oc))
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = self.red(torch.cat([x, skip], 1))
        return F.leaky_relu(self.res(x) + self.skip(x), 0.1)


class CausalConditioningLayer(nn.Module):
    """
    Blends features from causal parent decoders into the current task.
    Uses a learned gating mechanism (not hardcoded weights).

    parent_features: list of [B, feat_ch, H, W] tensors from parent decoders
    curr_features:   [B, feat_ch, H, W] from current decoder
    A_row:           [N_tasks] — row of causal adjacency for this task (parent weights)
    """
    def __init__(self, feat_ch: int, max_parents: int = 13, cond_dim: int = 32):
        super().__init__()
        self.feat_ch   = feat_ch
        self.cond_dim  = cond_dim

        # Project each parent feature to conditioning dim
        self.parent_proj = nn.Conv2d(feat_ch, cond_dim, 1, bias=False)
        # Gating network: decides how much parent signal to blend
        self.gate = nn.Sequential(
            nn.Conv2d(feat_ch + cond_dim, feat_ch, 1, bias=False),
            nn.BatchNorm2d(feat_ch),
            nn.Sigmoid(),
        )
        # Output: blend parent signal into current
        self.blend = nn.Sequential(
            nn.Conv2d(feat_ch + cond_dim, feat_ch, 1, bias=False),
            nn.BatchNorm2d(feat_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, curr: torch.Tensor,
                parent_feats: list,
                parent_weights: torch.Tensor) -> torch.Tensor:
        """
        curr:            [B, feat_ch, H, W]
        parent_feats:    list of [B, feat_ch, H, W]
        parent_weights:  [len(parent_feats)] — edge weights from DAG
        Returns:         [B, feat_ch, H, W]
        """
        if not parent_feats:
            return curr   # no parents → pass through unchanged

        # Weighted sum of parent projections
        cond = torch.zeros(curr.shape[0], self.cond_dim,
                           curr.shape[2], curr.shape[3],
                           device=curr.device)

        total_w = parent_weights.sum() + 1e-8
        for pf, pw in zip(parent_feats, parent_weights):
            # Upsample parent feature to match current spatial size
            pf_up = F.interpolate(pf, size=curr.shape[2:],
                                  mode='bilinear', align_corners=False)
            cond  = cond + (pw / total_w) * self.parent_proj(pf_up)

        # Gated blending
        combined = torch.cat([curr, cond], dim=1)
        gate     = self.gate(combined)
        out      = self.blend(combined)
        return curr * (1 - gate) + out * gate


# ─────────────────────────────────────────────────────────────
# Causally-Ordered Decoder (one instance per task)
# ─────────────────────────────────────────────────────────────

class CausalClsDecoder(nn.Module):
    def __init__(self, enc_ch, skips, dc, n_cls, cond_dim=32):
        super().__init__()
        self.d3   = DecBlock(enc_ch,   skips['s3'], dc)
        self.d2   = DecBlock(dc,       skips['s2'], dc//2)
        self.d1   = DecBlock(dc//2,    skips['s1'], dc//4)
        self.d0   = DecBlock(dc//4,    skips['s0'], dc//8)
        feat_ch   = dc // 8
        self.cond = CausalConditioningLayer(feat_ch, cond_dim=cond_dim)
        self.head = nn.Conv2d(feat_ch, n_cls, 1)
        self.feat_ch = feat_ch

    def forward(self, e, sk, tsz, parent_feats=None, parent_weights=None):
        x = self.d3(e, sk['s3']); x = self.d2(x, sk['s2'])
        x = self.d1(x, sk['s1']); x = self.d0(x, sk['s0'])
        x = F.interpolate(x, size=tsz, mode='bilinear', align_corners=False)
        if parent_feats:
            x = self.cond(x, parent_feats, parent_weights)
        return self.head(x), x   # (prediction, features_for_children)


class CausalRegDecoder(nn.Module):
    def __init__(self, enc_ch, skips, dc, cond_dim=32):
        super().__init__()
        self.d3   = DecBlock(enc_ch,   skips['s3'], dc)
        self.d2   = DecBlock(dc,       skips['s2'], dc//2)
        self.d1   = DecBlock(dc//2,    skips['s1'], dc//4)
        self.d0   = DecBlock(dc//4,    skips['s0'], dc//8)
        feat_ch   = dc // 8
        self.cond = CausalConditioningLayer(feat_ch, cond_dim=cond_dim)
        self.head = nn.Sequential(nn.Conv2d(feat_ch, 1, 1), nn.Sigmoid())
        self.feat_ch = feat_ch

    def forward(self, e, sk, tsz, parent_feats=None, parent_weights=None):
        x = self.d3(e, sk['s3']); x = self.d2(x, sk['s2'])
        x = self.d1(x, sk['s1']); x = self.d0(x, sk['s0'])
        x = F.interpolate(x, size=tsz, mode='bilinear', align_corners=False)
        if parent_feats:
            x = self.cond(x, parent_feats, parent_weights)
        return self.head(x), x   # (prediction, features_for_children)


# ─────────────────────────────────────────────────────────────
# Collection: runs all 13 decoders in causal order
# ─────────────────────────────────────────────────────────────

class CausalDecoderCollection(nn.Module):
    """
    Manages all 13 task decoders and runs them in topological order
    determined by the learned DAG (Module 1).

    Forward pass:
      1. Get topological order from DAG module
      2. For each task (in order): run decoder, collecting features
      3. Child tasks receive features from their parent decoders
    """
    def __init__(self, enc_ch, skips, dc, cond_dim=32):
        super().__init__()
        # Build decoders for all 13 tasks
        self.decoders = nn.ModuleDict()
        for v, n in CAT_VARIABLES.items():
            self.decoders[v] = CausalClsDecoder(enc_ch, skips, dc, n, cond_dim)
        for v in REG_VARIABLES:
            self.decoders[v] = CausalRegDecoder(enc_ch, skips, dc, cond_dim)

        self.all_vars  = ALL_VARS
        self.n_vars    = len(ALL_VARS)

    def forward(self, enc, skips, tsz, A_soft):
        """
        enc:    [B, enc_ch, H, W]
        skips:  dict of skip connections
        tsz:    target spatial size (H_in, W_in)
        A_soft: [13, 13] — soft causal adjacency from DAG module
        Returns:
            cls_preds: dict[var → [B, n_cls, H, W]]
            reg_preds: dict[var → [B, 1, H, W]]
        """
        # Topological order: run parents before children
        # Use simple degree-based approximation (full topo sort is in dag.py)
        in_degree = A_soft.sum(dim=0)        # [13] — how many parents each var has
        order     = torch.argsort(in_degree) # run low-degree (few parents) first
        order     = order.tolist()

        # Run decoders in order, storing intermediate features
        task_feats = {}   # var_name → feature map [B, feat_ch, H, W]
        cls_preds  = {}
        reg_preds  = {}

        for idx in order:
            var_name = self.all_vars[idx]

            # Gather parent features + weights
            parent_feats   = []
            parent_weights = []
            for parent_idx in range(self.n_vars):
                w = A_soft[parent_idx, idx]       # weight of edge parent→current
                if w.item() > 0.1 and self.all_vars[parent_idx] in task_feats:
                    parent_feats.append(task_feats[self.all_vars[parent_idx]])
                    parent_weights.append(w)

            pw_tensor = (torch.stack(parent_weights)
                         if parent_weights else None)

            dec    = self.decoders[var_name]
            pred, feat = dec(enc, skips, tsz, parent_feats, pw_tensor)

            task_feats[var_name] = feat.detach()  # detach to prevent gradient loops

            if var_name in CAT_VARIABLES:
                cls_preds[var_name] = pred
            else:
                reg_preds[var_name] = pred

        return cls_preds, reg_preds
