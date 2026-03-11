"""
models/causal/dag.py
=====================
MODULE 1 — Causal Discovery via Differentiable DAG Learning

Learns a DAG (Directed Acyclic Graph) over the 13 TAIGA forest variables.
Uses the NOTEARS approach (Zheng et al. 2018, NeurIPS):
  - Differentiable DAG constraint: h(W) = trace(exp(W⊙W)) - d = 0
  - Learns causal adjacency matrix W jointly with prediction
  - Exposes causal parent-child relationships for Module 3

13 TAIGA variables (nodes):
  0  fertility_class     (cat)
  1  soil_class          (cat)
  2  tree_species        (cat)
  3  basal_area          (reg)
  4  mean_dbh            (reg)
  5  stem_density        (reg)
  6  mean_height         (reg)
  7  pct_pine            (reg)
  8  pct_spruce          (reg)
  9  pct_birch           (reg)
  10 woody_biomass       (reg)
  11 leaf_area_index     (reg)
  12 eff_leaf_area_index (reg)

Expected learned causal structure (ecological knowledge):
  tree_species → pct_pine, pct_spruce, pct_birch
  fertility_class → basal_area, stem_density, mean_height
  mean_height → woody_biomass, leaf_area_index
  basal_area → woody_biomass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# All 13 variable names in order
ALL_VARIABLES = [
    "fertility_class", "soil_class", "tree_species",
    "basal_area", "mean_dbh", "stem_density", "mean_height",
    "pct_pine", "pct_spruce", "pct_birch",
    "woody_biomass", "leaf_area_index", "eff_leaf_area_index",
]

N_VARS = len(ALL_VARIABLES)   # 13

# ─────────────────────────────────────────────────────────────
# Domain Prior (ecological knowledge)
# ─────────────────────────────────────────────────────────────

def build_domain_prior() -> torch.Tensor:
    """
    Build A_prior [13×13] from ecological knowledge.
    A_prior[i,j] = 1 means variable i is a known cause of variable j.
    Used to initialize W_task and bias learning.
    """
    idx = {v: i for i, v in enumerate(ALL_VARIABLES)}
    A   = torch.zeros(N_VARS, N_VARS)

    # Tree species determines species percentages
    A[idx["tree_species"], idx["pct_pine"]]    = 1.0
    A[idx["tree_species"], idx["pct_spruce"]]  = 1.0
    A[idx["tree_species"], idx["pct_birch"]]   = 1.0

    # Site fertility drives stand structure
    A[idx["fertility_class"], idx["basal_area"]]    = 1.0
    A[idx["fertility_class"], idx["stem_density"]]  = 1.0
    A[idx["fertility_class"], idx["mean_height"]]   = 1.0
    A[idx["fertility_class"], idx["mean_dbh"]]      = 1.0

    # Soil class → species composition
    A[idx["soil_class"], idx["tree_species"]]  = 1.0
    A[idx["soil_class"], idx["pct_pine"]]      = 1.0

    # Stand structure → biomass/LAI
    A[idx["mean_height"],  idx["woody_biomass"]]      = 1.0
    A[idx["mean_height"],  idx["leaf_area_index"]]    = 1.0
    A[idx["basal_area"],   idx["woody_biomass"]]      = 1.0
    A[idx["basal_area"],   idx["leaf_area_index"]]    = 1.0
    A[idx["stem_density"], idx["basal_area"]]         = 1.0

    # LAI → effective LAI (clumping correction)
    A[idx["leaf_area_index"], idx["eff_leaf_area_index"]] = 1.0

    return A


class CausalDAGModule(nn.Module):
    """
    Learns a causal adjacency matrix W over the 13 TAIGA variables.
    Applies NOTEARS DAG constraint to enforce acyclicity.

    During forward pass, produces:
      - A_soft: sigmoid-gated soft adjacency [13, 13]
      - dag_penalty: scalar DAG constraint violation (pushed to 0)
    """
    def __init__(self, n_vars: int = N_VARS, use_prior: bool = True,
                 threshold: float = 0.5):
        super().__init__()
        self.n       = n_vars
        self.thresh  = threshold

        # Learnable raw weight matrix (log-odds of edge existence)
        self.W_raw = nn.Parameter(torch.zeros(n_vars, n_vars))
        nn.init.normal_(self.W_raw, mean=0, std=0.1)

        # Domain prior as bias (non-trainable)
        if use_prior:
            prior = build_domain_prior()
            # Scale prior to log-odds space: prior edges → +2.0, absent → 0.0
            self.register_buffer("prior_bias", prior * 2.0)
        else:
            self.register_buffer("prior_bias", torch.zeros(n_vars, n_vars))

        # No self-loops mask
        self.register_buffer("no_self", 1 - torch.eye(n_vars))

    def forward(self):
        """
        Returns:
            A_soft: [N, N] soft adjacency (sigmoid-gated, no self-loops)
            dag_penalty: scalar — NOTEARS constraint (minimise this)
            sparsity:    scalar — L1 of W (promotes sparse graph)
        """
        W = (self.W_raw + self.prior_bias) * self.no_self
        A_soft = torch.sigmoid(W)                    # [N, N] ∈ [0, 1]

        # NOTEARS acyclicity constraint: h(A) = trace(exp(A⊙A)) - d = 0
        # Using element-wise squared A for positive semi-definiteness
        A_sq          = A_soft * A_soft              # [N, N]
        exp_A         = torch.matrix_exp(A_sq)       # [N, N]
        dag_penalty   = (torch.trace(exp_A) - self.n) ** 2  # → 0 iff DAG

        sparsity = A_soft.abs().sum()

        return A_soft, dag_penalty, sparsity

    def get_hard_graph(self) -> torch.Tensor:
        """Return binary adjacency after thresholding (for visualization)."""
        A_soft, _, _ = self.forward()
        return (A_soft > self.thresh).float()

    def topological_order(self) -> list:
        """
        Compute topological sort of current causal graph.
        Returns list of variable indices in topological order.
        """
        A_hard = self.get_hard_graph().detach().cpu().numpy()
        visited = [False] * self.n
        order   = []

        def dfs(node):
            visited[node] = True
            for child in range(self.n):
                if A_hard[node, child] > 0.5 and not visited[child]:
                    dfs(child)
            order.append(node)

        for i in range(self.n):
            if not visited[i]:
                dfs(i)

        return order[::-1]   # reverse post-order = topological order

    def print_graph(self, threshold: float = 0.5):
        """Print discovered causal edges for inspection."""
        A_soft, dag_pen, _ = self.forward()
        A = A_soft.detach().cpu().numpy()
        print(f"\n[DAG] Causal graph (threshold={threshold}):")
        print(f"      DAG penalty: {dag_pen.item():.4f}  (0=perfect DAG)")
        edges = []
        for i in range(self.n):
            for j in range(self.n):
                if A[i, j] > threshold:
                    edges.append(f"  {ALL_VARIABLES[i]:<25} → {ALL_VARIABLES[j]} "
                                 f"(w={A[i,j]:.3f})")
        print("\n".join(edges) if edges else "  No edges above threshold")
        print()
