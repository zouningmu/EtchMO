import torch


def gaussian_kernel2d_torch(
    sigma_pix: float,
    truncate: float = 3.0,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    sigma_pix = float(max(sigma_pix, 1e-6))
    radius = int(truncate * sigma_pix + 0.5)
    k = 2 * radius + 1
    xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    ys = xs.view(-1, 1)
    grid_sq = xs * xs + ys * ys
    ker = torch.exp(-0.5 * grid_sq / (sigma_pix * sigma_pix))
    ker = ker / ker.sum()
    return ker.view(1, 1, k, k)