"""
losses/causal_loss.py
======================
All causal training losses added on top of the baseline multitask loss.

Total loss (annealed schedule):
  L = L_task                      (CE + MAE + uncertainty)
    + λ1 * L_IRM                  (IRM invariance penalty, anneal from epoch 20)
    + λ2 * L_DAG                  (NOTEARS DAG constraint, anneal from epoch 10)
    + λ3 * L_adv                  (adversarial environment discriminator)
    + λ4 * L_MI                   (HSIC independence)
    + λ5 * L_recon                (reconstruction — prevents info loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.causal.disentangle import irm_penalty, hsic_loss


class CausalLoss(nn.Module):
    """
    Wraps baseline MultitaskLoss and adds causal regularization terms.

    Annealing schedule (prevents causal losses from disrupting early training):
      Epochs 1–9:   baseline only
      Epochs 10–19: + L_DAG
      Epochs 20+:   + L_IRM + L_adv + L_MI
    """
    def __init__(self, base_loss_fn, n_envs: int = 7, cfg: dict = None):
        super().__init__()
        self.base_loss = base_loss_fn
        self.n_envs    = n_envs
        cc = (cfg or {}).get("causal", {})

        # Loss weights (tunable in config)
        self.lam_irm   = cc.get("lambda_irm",   0.1)
        self.lam_dag   = cc.get("lambda_dag",   1.0)
        self.lam_adv   = cc.get("lambda_adv",   0.1)
        self.lam_mi    = cc.get("lambda_mi",    0.05)
        self.lam_recon = cc.get("lambda_recon", 0.1)

        # Anneal start epochs
        self.dag_start = cc.get("dag_anneal_epoch", 10)
        self.irm_start = cc.get("irm_anneal_epoch", 20)

        # Environment discriminator CE
        self.disc_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        out:        dict,         # full model output dict
        cat_labels: dict,
        reg_labels: dict,
        env_ids:    torch.Tensor, # [B] flight strip IDs
        epoch:      int,
    ):
        """
        Returns (total_loss, details_dict)
        """
        details = {}

        # ── 1. Baseline multitask loss ─────────────────────────
        L_task, task_det = self.base_loss(
            out["cls"], out["reg"], cat_labels, reg_labels
        )
        details.update(task_det)
        total = L_task

        # ── 2. Reconstruction loss ─────────────────────────────
        # Ensures disentanglement doesn't discard information
        # Compare reconstructed vs encoder output (detached)
        enc_target = out["x_recon"].detach()
        L_recon = F.mse_loss(out["x_recon"], enc_target)
        total   = total + self.lam_recon * L_recon
        details["L_recon"] = L_recon.item()

        # ── 3. Adversarial environment loss ────────────────────
        # Discriminator should NOT be able to predict strip from Z_causal
        # GRL in the model already reverses gradient for the encoder side
        # Here we compute the discriminator classification loss for logging
        if env_ids is not None:
            L_adv = self.disc_loss(out["env_logits"], env_ids.long())
            total = total + self.lam_adv * L_adv
            details["L_adv"] = L_adv.item()
        else:
            details["L_adv"] = 0.0

        # ── 4. DAG constraint (anneal from epoch dag_start) ────
        if epoch >= self.dag_start:
            L_dag     = out["dag_penalty"]
            L_sparse  = out["dag_sparse"] * 0.01
            total     = total + self.lam_dag * (L_dag + L_sparse)
            details["L_dag"]    = L_dag.item()
            details["L_sparse"] = L_sparse.item()
        else:
            details["L_dag"] = 0.0

        # ── 5. IRM + HSIC (anneal from epoch irm_start) ───────
        if epoch >= self.irm_start and env_ids is not None:
            # IRM: compute per-environment task losses
            unique_envs = env_ids.unique()
            env_losses  = []
            for env in unique_envs:
                mask = (env_ids == env)
                if mask.sum() < 2:
                    continue
                # Get per-env task loss (cls only for efficiency)
                env_cls = {v: p[mask] for v, p in out["cls"].items()}
                env_cat = {v: t[mask] for v, t in cat_labels.items()}
                env_reg = {v: p[mask] for v, p in out["reg"].items()}
                env_rl  = {v: t[mask] for v, t in reg_labels.items()}
                L_env, _ = self.base_loss(env_cls, env_reg, env_cat, env_rl)
                env_losses.append(L_env)

            if len(env_losses) >= 2:
                L_irm = irm_penalty(env_losses)
                total = total + self.lam_irm * L_irm
                details["L_irm"] = L_irm.item()
            else:
                details["L_irm"] = 0.0

            # HSIC: independence between Z_causal and Z_spurious
            L_hsic = hsic_loss(out["z_causal"], out["z_spurious"])
            total  = total + self.lam_mi * L_hsic
            details["L_hsic"] = L_hsic.item()
        else:
            details["L_irm"]  = 0.0
            details["L_hsic"] = 0.0

        details["total"] = total.item()
        return total, details
