"""
PINNs network: single MLP with Fourier encoding and two output heads.

Architecture (v1.0):
  Input  : (r, x, t)                           [3]
  Trunk  : Fourier encoding → 8 × 128, tanh    [128]
  Field head : (u_r, u_x, p, Theta_f, Theta_dep)  [5]
  Interface head : masked (x, t) → r_int        [1]

The interface head receives only (x, t) — the r input is masked to zero —
encoding the physical inductive bias that r_int does not depend on r.
"""

import torch
import torch.nn as nn
import numpy as np
from config import NetworkConfig, CaseConfig


class FourierEncoding(nn.Module):
    """Random Fourier feature encoding.

    Maps each scalar v to [sin(2π B v), cos(2π B v)] where B is frozen random.
    Output dim = 2 * n_features.
    """

    def __init__(self, in_dim: int, n_features: int, sigma: float, seed: int = 42):
        super().__init__()
        rng = torch.Generator()
        rng.manual_seed(seed)
        B = torch.randn(in_dim, n_features, generator=rng) * sigma
        self.register_buffer("B", B)  # frozen, not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (..., in_dim)  →  (..., 2*n_features)"""
        proj = 2.0 * torch.pi * (x @ self.B)          # (..., n_features)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


def _mlp_block(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.Tanh())


class PINNSolidification(nn.Module):
    """
    Single network for 2D axisymmetric pipe solidification.

    Inputs  : r, x, t  (each scalar, shape [N] or [N,1])
    Outputs :
        u_r      — radial velocity (non-dim)
        u_x      — axial velocity (non-dim)
        p        — pressure (non-dim)
        Theta_f  — fluid temperature in [0, 1]
        Theta_dep — deposit temperature in [0, 1]
        r_int    — interface position in [0, r_w]
    """

    def __init__(self, net_cfg: NetworkConfig, case_cfg: CaseConfig, seed: int = 42):
        super().__init__()
        self.case = case_cfg

        # ── Fourier encoding (applied to full 3-input and 2-input separately) ─
        enc_out = 2 * net_cfg.fourier_features
        self.fourier_3d = FourierEncoding(3, net_cfg.fourier_features, net_cfg.fourier_sigma, seed)
        self.fourier_2d = FourierEncoding(2, net_cfg.fourier_features, net_cfg.fourier_sigma, seed + 1)

        # ── Shared trunk: Fourier → hidden layers ─────────────────────────────
        trunk_layers = [_mlp_block(enc_out, net_cfg.n_hidden_units)]
        for _ in range(net_cfg.n_hidden_layers - 1):
            trunk_layers.append(_mlp_block(net_cfg.n_hidden_units, net_cfg.n_hidden_units))
        self.trunk = nn.Sequential(*trunk_layers)

        # ── Interface sub-trunk: operates on (x, t) only ─────────────────────
        int_trunk_layers = [_mlp_block(enc_out, net_cfg.n_hidden_units)]
        for _ in range(net_cfg.n_hidden_layers // 2 - 1):
            int_trunk_layers.append(_mlp_block(net_cfg.n_hidden_units, net_cfg.n_hidden_units))
        self.int_trunk = nn.Sequential(*int_trunk_layers)

        # ── Field head: (u_r, u_x, p, Theta_f, Theta_dep) ───────────────────
        self.field_head = nn.Linear(net_cfg.n_hidden_units, 5)

        # ── Interface head: → r_int ────────────────────────────────────────
        self.int_head = nn.Linear(net_cfg.n_hidden_units, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, r: torch.Tensor, x: torch.Tensor, t: torch.Tensor):
        """
        r, x, t : shape [N]  (flat tensors, requires_grad=True for PDE residuals)

        Returns dict with keys:
            u_r, u_x, p, Theta_f, Theta_dep  — shape [N]
            r_int                              — shape [N]
        """
        r = r.unsqueeze(-1)   # [N, 1]
        x = x.unsqueeze(-1)
        t = t.unsqueeze(-1)

        # Field trunk: all three inputs
        inp_3d = torch.cat([r, x, t], dim=-1)          # [N, 3]
        enc_3d = self.fourier_3d(inp_3d)                # [N, 2F]
        features = self.trunk(enc_3d)                   # [N, H]
        raw = self.field_head(features)                 # [N, 5]

        u_r = raw[:, 0]
        u_x = raw[:, 1]
        p   = raw[:, 2]
        Theta_f   = torch.sigmoid(raw[:, 3])   # constrained to (0, 1)
        Theta_dep = torch.sigmoid(raw[:, 4])

        # Interface trunk: (x, t) only — r is NOT used
        inp_2d = torch.cat([x, t], dim=-1)              # [N, 2]
        enc_2d = self.fourier_2d(inp_2d)                # [N, 2F]
        int_feat = self.int_trunk(enc_2d)               # [N, H]
        r_int = torch.sigmoid(self.int_head(int_feat)).squeeze(-1) * self.case.r_w

        return {
            "u_r":       u_r,
            "u_x":       u_x,
            "p":         p,
            "Theta_f":   Theta_f,
            "Theta_dep": Theta_dep,
            "r_int":     r_int,
        }

    def smooth_heaviside(self, r: torch.Tensor, r_int: torch.Tensor,
                          epsilon: float) -> torch.Tensor:
        """H_eps(r, r_int): ~0 in fluid (r < r_int), ~1 in deposit (r > r_int)."""
        return 0.5 * (1.0 + torch.tanh((r - r_int) / epsilon))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from config import cfg

    torch.manual_seed(cfg.seed)
    model = PINNSolidification(cfg.network, cfg.case, seed=cfg.seed)
    print(f"Total trainable parameters: {count_parameters(model):,}")

    # ── Smoke test ─────────────────────────────────────────────────────────────
    N = 100
    r = torch.rand(N, requires_grad=True)
    x = torch.rand(N) * cfg.case.L
    x.requires_grad_(True)
    t = torch.rand(N, requires_grad=True)

    out = model(r, x, t)

    print("\nOutput shapes and ranges:")
    for k, v in out.items():
        print(f"  {k:12s}: shape={tuple(v.shape)}  min={v.min().item():.4f}  max={v.max().item():.4f}")

    # Verify r_int has zero gradient w.r.t. r (key inductive bias check)
    loss = out["r_int"].sum()
    loss.backward()
    assert r.grad is None or r.grad.abs().max().item() < 1e-12, \
        "FAIL: r_int should not depend on r"
    print("\n[PASS] r_int gradient w.r.t. r is zero — interface head correctly masked.")
