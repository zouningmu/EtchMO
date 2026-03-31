import torch
import torch.nn.functional as F


def binary_dilate(mask01_4d: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return mask01_4d > 0.5
    k = 2 * r + 1
    d = F.max_pool2d(mask01_4d, kernel_size=k, stride=1, padding=r)
    return d > 0.5


@torch.no_grad()
def bbox_from_bool(mask_bool_4d: torch.Tensor):
    ys, xs = torch.where(mask_bool_4d[0, 0])
    if ys.numel() == 0:
        return None
    y0 = int(ys.min().item()); y1 = int(ys.max().item()) + 1
    x0 = int(xs.min().item()); x1 = int(xs.max().item()) + 1
    return y0, y1, x0, x1


def conv2d_edge_only_full(
    x4: torch.Tensor,
    k4: torch.Tensor,
    edge_bin_4d: torch.Tensor,
    pad: int,
    band_r: int = 0,
) -> torch.Tensor:
    B, _, H, W = x4.shape
    device, dtype = x4.device, x4.dtype

    keep = edge_bin_4d
    if keep.dtype != torch.bool:
        keep = keep > 0.5
    if band_r > 0:
        keep = binary_dilate(keep.float(), r=band_r)

    support = binary_dilate(keep.float(), r=pad)
    bbox = bbox_from_bool(support)
    if bbox is None:
        return torch.zeros((B, 1, H, W), device=device, dtype=dtype)

    y0, y1, x0, x1 = bbox

    x_roi = x4[:, :, y0:y1, x0:x1]
    y_roi = F.conv2d(x_roi, k4, padding=pad)

    keep_roi = keep[:, :, y0:y1, x0:x1].to(dtype)
    y_roi = y_roi * keep_roi

    y_full = torch.zeros((B, 1, H, W), device=device, dtype=dtype)
    y_full[:, :, y0:y1, x0:x1] = y_roi
    return y_full