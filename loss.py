"""
losses/loss.py — All multitask loss functions for TAIGA.
Focal loss, MAE, Uncertainty weighting (paper Eq. 15-21).
"""
import torch, torch.nn as nn, torch.nn.functional as F
from data.envi_reader import CAT_VARIABLES, REG_VARIABLES


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weights=None):
        super().__init__()
        self.a, self.g, self.w = alpha, gamma, weights

    def forward(self, logits, targets):
        B,C,H,W = logits.shape
        lsm = F.log_softmax(logits, 1)
        pt  = torch.exp(lsm).gather(1, targets.unsqueeze(1).long()).squeeze(1)
        lpt = lsm.gather(1, targets.unsqueeze(1).long()).squeeze(1)
        loss = -self.a * (1-pt)**self.g * lpt
        if self.w is not None:
            loss = loss * self.w.to(logits.device)[targets.long()]
        return loss.mean()


class UncertaintyLoss(nn.Module):
    def __init__(self, n_cls, n_reg):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_cls + n_reg))
        self.n_cls    = n_cls

    def forward(self, cls_losses, reg_losses):
        total, weights = torch.tensor(0., device=self.log_vars.device), {}
        all_items = list(cls_losses.items()) + list(reg_losses.items())
        for i, (k, L) in enumerate(all_items):
            lv = self.log_vars[i]
            s2 = torch.exp(-lv)
            w  = 0.5*s2*L + 0.5*lv if i >= self.n_cls else s2*L + 0.5*lv
            total = total + w
            weights[k] = s2.item()
        return total, weights


class MultitaskLoss(nn.Module):
    def __init__(self, class_weights, cfg):
        super().__init__()
        lc = cfg["loss"]
        self.cls_fns = nn.ModuleDict({
            v: FocalLoss(lc["focal_alpha"], lc["focal_gamma"],
                         class_weights.get(v))
            for v in CAT_VARIABLES
        })
        self.balancer = UncertaintyLoss(len(CAT_VARIABLES), len(REG_VARIABLES))

    def forward(self, cls_preds, reg_preds, cat_labels, reg_labels):
        # Classification losses
        cls_L = {}
        for v, fn in self.cls_fns.items():
            p = cls_preds[v]            # [B, n_cls, H, W]
            t = cat_labels[v]           # [B]
            B,C,H,W = p.shape
            t_map = t.view(B,1,1).expand(B,H,W)
            cls_L[v] = fn(p, t_map)

        # Regression losses (MAE on center pixel)
        reg_L = {}
        for v in REG_VARIABLES:
            p = reg_preds[v]            # [B, 1, H, W]
            H,W = p.shape[2:]
            center = p[:,0,H//2,W//2]  # [B]
            reg_L[v] = F.l1_loss(center, reg_labels[v].float())

        total, wts = self.balancer(cls_L, reg_L)
        details = {f"cls/{k}":v.item() for k,v in cls_L.items()}
        details.update({f"reg/{k}":v.item() for k,v in reg_L.items()})
        details["total"] = total.item()
        return total, details
