"""
PINN network for the active sharp-interface Case B model.

The field branch predicts ``u_r, u_x, p, Theta_f, Theta_dep`` from
``(r, x, tau)``. The interface branch predicts ``r_int`` from ``(x, tau)``,
so ``dr_int/dr`` is zero by construction.
"""

import torch
import torch.nn as nn

from config import CaseConfig, NetworkConfig


class FourierEncoding(nn.Module):
    """Frozen random Fourier feature encoding."""

    def __init__(self, in_dim: int, n_features: int, sigma: float, seed: int = 42):
        super().__init__()
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.register_buffer("B", torch.randn(in_dim, n_features, generator=rng) * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * torch.pi * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


def _mlp_block(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.Tanh())


class PINNSolidification(nn.Module):
    """Two-branch PINN for 2D axisymmetric pipe solidification."""

    def __init__(self, net_cfg: NetworkConfig, case_cfg: CaseConfig, seed: int = 42):
        super().__init__()
        self.case = case_cfg

        enc_out = 2 * net_cfg.fourier_features
        self.fourier_3d = FourierEncoding(
            3,
            net_cfg.fourier_features,
            net_cfg.fourier_sigma,
            seed,
        )
        self.fourier_2d = FourierEncoding(
            2,
            net_cfg.fourier_features,
            net_cfg.fourier_sigma,
            seed + 1,
        )

        trunk_layers = [_mlp_block(enc_out, net_cfg.n_hidden_units)]
        for _ in range(net_cfg.n_hidden_layers - 1):
            trunk_layers.append(_mlp_block(net_cfg.n_hidden_units, net_cfg.n_hidden_units))
        self.trunk = nn.Sequential(*trunk_layers)

        int_trunk_layers = [_mlp_block(enc_out, net_cfg.n_hidden_units)]
        for _ in range(net_cfg.n_hidden_layers // 2 - 1):
            int_trunk_layers.append(_mlp_block(net_cfg.n_hidden_units, net_cfg.n_hidden_units))
        self.int_trunk = nn.Sequential(*int_trunk_layers)

        self.field_head = nn.Linear(net_cfg.n_hidden_units, 5)
        self.int_head = nn.Linear(net_cfg.n_hidden_units, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def _normalise_inputs(
        self,
        r: torch.Tensor,
        x: torch.Tensor,
        tau: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            r / max(self.case.r_w, 1e-12),
            x / max(self.case.L, 1e-12),
            tau / max(self.case.tau_end, 1e-12),
        )

    def forward(self, r: torch.Tensor, x: torch.Tensor, tau: torch.Tensor) -> dict[str, torch.Tensor]:
        r = r.unsqueeze(-1)
        x = x.unsqueeze(-1)
        tau = tau.unsqueeze(-1)
        r_n, x_n, tau_n = self._normalise_inputs(r, x, tau)

        field_input = torch.cat([r_n, x_n, tau_n], dim=-1)
        field_features = self.trunk(self.fourier_3d(field_input))
        raw = self.field_head(field_features)

        theta_max = max(
            self.case.Theta_in,
            self.case.Theta_solidus,
            self.case.Theta_wall,
            1.0,
        )

        interface_input = torch.cat([x_n, tau_n], dim=-1)
        interface_features = self.int_trunk(self.fourier_2d(interface_input))

        return {
            "u_r": raw[:, 0],
            "u_x": raw[:, 1],
            "p": raw[:, 2],
            "Theta_f": theta_max * torch.sigmoid(raw[:, 3]),
            "Theta_dep": theta_max * torch.sigmoid(raw[:, 4]),
            "r_int": torch.sigmoid(self.int_head(interface_features)).squeeze(-1) * self.case.r_w,
        }

    def smooth_heaviside(
        self,
        r: torch.Tensor,
        r_int: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        """Smooth indicator, near 0 in fluid and near 1 in deposit."""
        return 0.5 * (1.0 + torch.tanh((r - r_int) / epsilon))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
