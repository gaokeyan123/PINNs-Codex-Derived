"""
PINNs network: one neural model with two branches.

The field branch predicts local flow and temperature fields:
    (u_r, u_x, p, Theta_f, Theta_dep) = F(r, x, tau)

The interface branch predicts the moving phase boundary:
    r_int = G(x, tau)

The interface branch intentionally does not receive r.  That makes
partial r_int / partial r exactly zero by construction, matching the
physical idea that the interface radius is a single value at each x,tau.
"""

import torch  # Main tensor/autograd library used by the PINN.
import torch.nn as nn  # PyTorch neural-network layers and base Module class.
import numpy as np  # Currently unused; kept because older experiments imported it here.
from config import NetworkConfig, CaseConfig  # Type hints for architecture and case settings.


class FourierEncoding(nn.Module):  # Random Fourier feature layer for spectral enrichment.
    """Random Fourier feature encoding."""

    def __init__(self, in_dim: int, n_features: int, sigma: float, seed: int = 42):  # Build frozen random frequencies.
        super().__init__()  # Initialize nn.Module internals.
        rng = torch.Generator()  # Create a local random generator so frequencies are reproducible.
        rng.manual_seed(seed)  # Fix the random seed for this Fourier matrix.
        B = torch.randn(in_dim, n_features, generator=rng) * sigma  # Draw random frequencies for each input coordinate.
        self.register_buffer("B", B)  # Store B on the model, but do not train it as a parameter.

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Convert coordinates to sin/cos Fourier features.
        """Map x with shape (..., in_dim) to shape (..., 2*n_features)."""
        proj = 2.0 * torch.pi * (x @ self.B)  # Project inputs onto frozen random frequencies.
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # Return both sine and cosine channels.


def _mlp_block(in_dim: int, out_dim: int) -> nn.Sequential:  # Helper for one hidden MLP layer.
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.Tanh())  # Linear transform followed by smooth tanh activation.


class PINNSolidification(nn.Module):  # Main neural network used by training and postprocessing.
    """Single network for 2D axisymmetric pipe solidification."""

    def __init__(self, net_cfg: NetworkConfig, case_cfg: CaseConfig, seed: int = 42):  # Create all layers.
        super().__init__()  # Initialize nn.Module internals.
        self.case = case_cfg  # Save case parameters such as r_w, L, tau_end, and temperature scale.

        enc_out = 2 * net_cfg.fourier_features  # FourierEncoding outputs sin and cos, so channels double.
        self.fourier_3d = FourierEncoding(  # Encoder for full field inputs (r, x, tau).
            3,  # Field branch uses three coordinates.
            net_cfg.fourier_features,  # Number of random frequencies.
            net_cfg.fourier_sigma,  # Frequency scale.
            seed,  # Seed for reproducible field-branch features.
        )
        self.fourier_2d = FourierEncoding(  # Encoder for interface inputs (x, tau).
            2,  # Interface branch uses only two coordinates.
            net_cfg.fourier_features,  # Number of random frequencies.
            net_cfg.fourier_sigma,  # Frequency scale.
            seed + 1,  # Different seed so the interface branch has its own features.
        )

        trunk_layers = [_mlp_block(enc_out, net_cfg.n_hidden_units)]  # First field hidden layer.
        for _ in range(net_cfg.n_hidden_layers - 1):  # Add the remaining field hidden layers.
            trunk_layers.append(  # Append one more tanh MLP block.
                _mlp_block(net_cfg.n_hidden_units, net_cfg.n_hidden_units)  # Hidden-to-hidden layer.
            )
        self.trunk = nn.Sequential(*trunk_layers)  # Field trunk maps Fourier features to hidden state.

        int_trunk_layers = [_mlp_block(enc_out, net_cfg.n_hidden_units)]  # First interface hidden layer.
        for _ in range(net_cfg.n_hidden_layers // 2 - 1):  # Interface trunk is half as deep as field trunk.
            int_trunk_layers.append(  # Append one more interface MLP block.
                _mlp_block(net_cfg.n_hidden_units, net_cfg.n_hidden_units)  # Hidden-to-hidden layer.
            )
        self.int_trunk = nn.Sequential(*int_trunk_layers)  # Interface trunk maps x,tau features to hidden state.

        self.field_head = nn.Linear(net_cfg.n_hidden_units, 5)  # Predict u_r, u_x, p, Theta_f, Theta_dep.
        self.int_head = nn.Linear(net_cfg.n_hidden_units, 1)  # Predict one scalar: r_int.

        self._init_weights()  # Apply deterministic Xavier initialization to linear layers.

    def _init_weights(self):  # Initialize trainable neural-network weights.
        for m in self.modules():  # Loop over every submodule inside this model.
            if isinstance(m, nn.Linear):  # Only initialize dense Linear layers.
                nn.init.xavier_normal_(m.weight)  # Xavier normal keeps activations reasonably scaled.
                nn.init.zeros_(m.bias)  # Start all biases from zero.

    def _normalise_inputs(  # Scale physical non-dimensional coordinates before Fourier encoding.
        self,  # Current model instance.
        r: torch.Tensor,  # Radial coordinate tensor.
        x: torch.Tensor,  # Axial coordinate tensor.
        t: torch.Tensor,  # Tau coordinate tensor.
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # Return normalized r, x, and tau.
        """Scale each coordinate to roughly [0, 1] before Fourier encoding."""
        r_n = r / max(self.case.r_w, 1e-12)  # Normalize radius by pipe wall radius.
        x_n = x / max(self.case.L, 1e-12)  # Normalize axial position by pipe length.
        t_n = t / max(self.case.tau_end, 1e-12)  # Normalize tau by final case tau.
        return r_n, x_n, t_n  # Return normalized coordinates for both network branches.

    def forward(self, r: torch.Tensor, x: torch.Tensor, t: torch.Tensor):  # Evaluate the PINN at points.
        """Run a forward pass for flat tensors r, x, and tau with shape [N]."""
        r = r.unsqueeze(-1)  # Change shape from [N] to [N, 1] for concatenation.
        x = x.unsqueeze(-1)  # Change shape from [N] to [N, 1] for concatenation.
        t = t.unsqueeze(-1)  # Change shape from [N] to [N, 1] for concatenation.
        r_n, x_n, t_n = self._normalise_inputs(r, x, t)  # Normalize coordinates before Fourier features.

        inp_3d = torch.cat([r_n, x_n, t_n], dim=-1)  # Field branch input: [r, x, tau], shape [N, 3].
        enc_3d = self.fourier_3d(inp_3d)  # Fourier-encode field coordinates, shape [N, 2F].
        features = self.trunk(enc_3d)  # Run encoded coordinates through the field MLP trunk.
        raw = self.field_head(features)  # Produce five raw field outputs, shape [N, 5].

        u_r = raw[:, 0]  # Radial velocity output is left unbounded.
        u_x = raw[:, 1]  # Axial velocity output is left unbounded.
        p = raw[:, 2]  # Pressure output is left unbounded because pressure has gauge freedom.
        theta_max = max(  # Upper bound used by sigmoid temperature outputs.
            self.case.Theta_in,  # Inlet temperature may be the largest temperature.
            self.case.Theta_solidus,  # Interface/solidus temperature may be largest in other cases.
            self.case.Theta_wall,  # Wall temperature may be largest in no-cooling tests.
            1.0,  # Ensure the temperature range is never smaller than one.
        )
        Theta_f = theta_max * torch.sigmoid(raw[:, 3])  # Fluid temperature bounded to [0, theta_max].
        Theta_dep = theta_max * torch.sigmoid(raw[:, 4])  # Deposit temperature bounded to [0, theta_max].

        inp_2d = torch.cat([x_n, t_n], dim=-1)  # Interface branch input: [x, tau], shape [N, 2].
        enc_2d = self.fourier_2d(inp_2d)  # Fourier-encode interface coordinates, shape [N, 2F].
        int_feat = self.int_trunk(enc_2d)  # Run encoded x,tau through the interface MLP trunk.
        r_int = torch.sigmoid(self.int_head(int_feat)).squeeze(-1) * self.case.r_w  # Bound interface to [0, r_w].

        return {  # Return all predicted quantities in a named dictionary.
            "u_r": u_r,  # Radial velocity field.
            "u_x": u_x,  # Axial velocity field.
            "p": p,  # Pressure field.
            "Theta_f": Theta_f,  # Fluid temperature field.
            "Theta_dep": Theta_dep,  # Deposit temperature field.
            "r_int": r_int,  # Interface radius depending only on x and tau.
        }

    def smooth_heaviside(  # Smooth region indicator for fluid/deposit blending.
        self,  # Current model instance.
        r: torch.Tensor,  # Radial coordinate tensor.
        r_int: torch.Tensor,  # Predicted interface radius tensor.
        epsilon: float,  # Width of the smooth transition around the interface.
    ) -> torch.Tensor:  # Return H near 0 in fluid and near 1 in deposit.
        """H_eps(r, r_int): about 0 in fluid (r < r_int), about 1 in deposit (r > r_int)."""
        return 0.5 * (1.0 + torch.tanh((r - r_int) / epsilon))  # Smooth approximation of a step function.


def count_parameters(model: nn.Module) -> int:  # Utility for reporting model size.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # Count trainable scalar parameters.


if __name__ == "__main__":  # Run this block only when executing network.py directly.
    from config import cfg  # Load the default project config for a local smoke test.

    torch.manual_seed(cfg.seed)  # Make this standalone smoke test reproducible.
    model = PINNSolidification(cfg.network, cfg.case, seed=cfg.seed)  # Build the PINN model.
    print(f"Total trainable parameters: {count_parameters(model):,}")  # Display model size.

    N = 100  # Number of random points for the smoke test.
    r = torch.rand(N, requires_grad=True)  # Random radial locations with autograd enabled.
    x = torch.rand(N) * cfg.case.L  # Random axial locations over the pipe length.
    x.requires_grad_(True)  # Enable derivatives with respect to x.
    t = (torch.rand(N) * cfg.case.tau_end).requires_grad_(True)  # Random tau values with autograd enabled.

    out = model(r, x, t)  # Evaluate all model outputs at the random points.

    print("\nOutput shapes and ranges:")  # Header for simple output inspection.
    for k, v in out.items():  # Loop through each predicted variable.
        print(f"  {k:12s}: shape={tuple(v.shape)}  min={v.min().item():.4f}  max={v.max().item():.4f}")  # Print summary.

    loss = out["r_int"].sum()  # Scalar expression using only the interface output.
    loss.backward()  # Backpropagate to check whether r_int depends on r.
    assert r.grad is None or r.grad.abs().max().item() < 1e-12, "FAIL: r_int should not depend on r"  # Enforce bias.
    print("\n[PASS] r_int gradient w.r.t. r is zero: interface head uses only x and tau.")  # Report success.
