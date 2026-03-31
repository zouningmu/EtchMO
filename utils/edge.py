import torch
import torch.nn.functional as F


def sobel_edge_weight(
    mask_soft_4d: torch.Tensor,
    eps: float = 1e-12,
    edge_tau: float = 1.0,
    edge_thr: float = 0.05,
) -> torch.Tensor:
    device = mask_soft_4d.device
    dtype = mask_soft_4d.dtype

    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [0,  0,  0],
                       [1,  2,  1]], device=device, dtype=dtype).view(1, 1, 3, 3)

    gx = F.conv2d(mask_soft_4d, kx, padding=1)
    gy = F.conv2d(mask_soft_4d, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + eps)

    return torch.sigmoid((mag - edge_thr) / max(edge_tau, 1e-6))