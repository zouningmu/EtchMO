import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

def _sobel_kernels(device, dtype):
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0],
         [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]],
        device=device, dtype=dtype
    ).view(1, 1, 3, 3)

    ky = torch.tensor(
        [[-1.0, -2.0, -1.0],
         [0.0,  0.0,  0.0],
         [1.0,  2.0,  1.0]],
        device=device, dtype=dtype
    ).view(1, 1, 3, 3)

    return kx, ky


def grad_magnitude(phi: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if phi.dim() != 4:
        raise ValueError(f"phi must be 4D (B,1,H,W), got {phi.dim()}D")

    kx, ky = _sobel_kernels(phi.device, phi.dtype)
    dx = F.conv2d(phi, kx, padding=1) / 8.0
    dy = F.conv2d(phi, ky, padding=1) / 8.0
    return torch.sqrt(dx * dx + dy * dy + eps)


def smooth_sign(phi0: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    if eps <= 0:
        raise ValueError("eps must be > 0")
    return phi0 / torch.sqrt(phi0 * phi0 + eps * eps)


def reinit_levelset_dynamic(
    phi0: torch.Tensor,
    dt: float = 0.3,
    sign_eps: float = 1.0,
    grad_eps: float = 1e-6,
    narrow_band: float = None,
    target_grad_range: tuple = (0.9, 1.1),
    tol: float = 0.01,
    max_iters: int = 30,
    verbose: bool = True,
    stable_iters: int = 2,
    dt_halve_every: int = 15,
    update_clip: float = 0.5,
):
    if max_iters <= 0:
        raise ValueError("max_iters must be > 0")
    if stable_iters <= 0:
        raise ValueError("stable_iters must be > 0")
    if dt <= 0:
        raise ValueError("dt must be > 0")

    # shape normalize: (H,W) / (B,H,W) / (B,1,H,W)
    if phi0.dim() == 2:
        phi0 = phi0[None, None, ...]
    elif phi0.dim() == 3:
        phi0 = phi0[:, None, ...]
    elif phi0.dim() == 4:
        if phi0.size(1) != 1:
            raise ValueError(f"phi0 channel must be 1, got shape={tuple(phi0.shape)}")
    else:
        raise ValueError("phi0 must be 2D/3D/4D tensor")

    phi = phi0.clone()
    s = smooth_sign(phi0, eps=sign_eps)

    band_mask = None
    if narrow_band is not None:
        band_mask = (phi0.abs() < float(narrow_band))  # bool

    prev_grad_mean = None
    converge_count = 0

    cur_dt = float(dt)

    for it in range(max_iters):
        if dt_halve_every and dt_halve_every > 0 and (it > 0) and (it % dt_halve_every == 0):
            cur_dt *= 0.5

        g = grad_magnitude(phi, eps=grad_eps)

        update = s * (1.0 - g) * torch.clamp(1.0 - 0.2 * torch.abs(1.0 - g), 0.6, 1.0)
        update = torch.clamp(update, -update_clip, update_clip)

        if band_mask is not None:
            update = update.masked_fill(~band_mask, 0.0)

        phi = phi + cur_dt * update

        # convergence stats
        g_det = g.detach()
        if band_mask is not None and band_mask.any():
            curr_grad_mean = g_det[band_mask].mean().item()
        else:
            curr_grad_mean = g_det.mean().item()

        in_range = (target_grad_range[0] <= curr_grad_mean <= target_grad_range[1])

        if prev_grad_mean is not None:
            fluctuate_less = abs(curr_grad_mean - prev_grad_mean) < tol
            if in_range and fluctuate_less:
                converge_count += 1
            else:
                converge_count = 0

        if converge_count >= stable_iters:
            break

        prev_grad_mean = curr_grad_mean

    return phi, band_mask


