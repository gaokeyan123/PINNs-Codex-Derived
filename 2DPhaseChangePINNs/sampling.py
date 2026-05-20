"""
Collocation point generation for PINN training.

Each returned point set contains flat tensors named ``r``, ``x``, and ``tau``
with ``requires_grad=True`` so residual functions can use autograd.
"""

import torch

from config import Config

_AXIS_R = 1e-4


def _lhs(N: int, d: int, seed: int = 0) -> torch.Tensor:
    """Latin Hypercube sample in [0, 1]^d with shape [N, d]."""
    gen = torch.Generator()
    gen.manual_seed(seed)

    edges = torch.arange(N, dtype=torch.float32) / N
    out = torch.empty(N, d)
    for axis in range(d):
        perm = torch.randperm(N, generator=gen)
        offsets = torch.rand(N, generator=gen) / N
        out[:, axis] = edges[perm] + offsets
    return out


def _pack(r: torch.Tensor, x: torch.Tensor, tau: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "r": r.detach().requires_grad_(True),
        "x": x.detach().requires_grad_(True),
        "tau": tau.detach().requires_grad_(True),
    }


def sample_interior(N: int, cfg: Config, seed: int = 0) -> dict[str, torch.Tensor]:
    """Interior points in r, x, and tau."""
    s = _lhs(N, 3, seed)
    r_min = max(0.0, float(cfg.sampling.r_min_interior))
    if r_min >= cfg.case.r_w:
        raise ValueError(
            f"cfg.sampling.r_min_interior must be smaller than r_w={cfg.case.r_w}"
        )
    r = r_min + s[:, 0] * (cfg.case.r_w - r_min)
    x = s[:, 1] * cfg.case.L
    tau = s[:, 2] * cfg.case.tau_end
    return _pack(r, x, tau)


def sample_wall(N: int, cfg: Config, seed: int = 0) -> dict[str, torch.Tensor]:
    s = _lhs(N, 2, seed)
    r = torch.full((N,), cfg.case.r_w)
    x = s[:, 0] * cfg.case.L
    tau = s[:, 1] * cfg.case.tau_end
    return _pack(r, x, tau)


def sample_inlet(N: int, cfg: Config, seed: int = 0) -> dict[str, torch.Tensor]:
    s = _lhs(N, 2, seed)
    r = s[:, 0] * cfg.case.r_w
    x = torch.zeros(N)
    tau = s[:, 1] * cfg.case.tau_end
    return _pack(r, x, tau)


def sample_outlet(N: int, cfg: Config, seed: int = 0) -> dict[str, torch.Tensor]:
    s = _lhs(N, 2, seed)
    r = s[:, 0] * cfg.case.r_w
    x = torch.full((N,), cfg.case.L)
    tau = s[:, 1] * cfg.case.tau_end
    return _pack(r, x, tau)


def sample_axis(N: int, cfg: Config, seed: int = 0) -> dict[str, torch.Tensor]:
    s = _lhs(N, 2, seed)
    r = torch.full((N,), _AXIS_R)
    x = s[:, 0] * cfg.case.L
    tau = s[:, 1] * cfg.case.tau_end
    return _pack(r, x, tau)


def sample_ic(N: int, cfg: Config, seed: int = 0) -> dict[str, torch.Tensor]:
    s = _lhs(N, 2, seed)
    r = s[:, 0] * cfg.case.r_w
    x = s[:, 1] * cfg.case.L
    tau = torch.zeros(N)
    return _pack(r, x, tau)


def sample_interface(model, N: int, cfg: Config, seed: int = 0) -> dict[str, torch.Tensor]:
    """Adaptive interface points with r sampled from the current interface head."""
    device = next(model.parameters()).device
    s = _lhs(N, 2, seed).to(device)
    x_samp = s[:, 0] * cfg.case.L
    tau_samp = s[:, 1] * cfg.case.tau_end

    with torch.no_grad():
        r_dummy = torch.zeros(N, device=device)
        r_int = model(r_dummy, x_samp, tau_samp)["r_int"].detach().clamp(0.0, cfg.case.r_w)

    return _pack(r_int.clone(), x_samp.clone(), tau_samp.clone())


def resample_interface(model, N: int, cfg: Config, seed: int = 0) -> dict[str, torch.Tensor]:
    return sample_interface(model, N, cfg, seed)


def build_batch(model, cfg: Config, seed: int = 0) -> dict[str, dict[str, torch.Tensor]]:
    sc = cfg.sampling
    return {
        "interior": sample_interior(sc.N_interior, cfg, seed + 0),
        "interface": sample_interface(model, sc.N_interface, cfg, seed + 1),
        "wall": sample_wall(sc.N_boundary, cfg, seed + 2),
        "inlet": sample_inlet(sc.N_boundary, cfg, seed + 3),
        "outlet": sample_outlet(sc.N_boundary, cfg, seed + 4),
        "axis": sample_axis(sc.N_boundary, cfg, seed + 5),
        "ic": sample_ic(sc.N_ic, cfg, seed + 6),
    }
