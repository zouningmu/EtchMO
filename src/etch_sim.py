import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn

from utils.mask import to_4d_mask, clamp01
from utils.edge import sobel_edge_weight
from utils.bias import build_kernels, compute_features_and_raw, extend_bias
from utils.redistancing import reinit_levelset_dynamic

import numpy as np
from PIL import Image
import torch.nn.functional as F

class DiffVEBEtch2D(nn.Module):
    def __init__(
        self,
        es_D1: float = 20.0,
        es_D2: float = 50.0,
        es_D3: float = 50.0,
        es_D4: float = 100.0,
        ext_sigma: float = 20.0,
        max_bias_nm: float = 20.0,
        phi_tau: float = 1.0,
        render_tau: float = 1.0,
        edge_tau: float = 1.0,
        edge_thr: float = 0.05,
        truncate: float = 1.0,
        band_r: int = 5,
        edge_bin_thr: float = 0.5,
        # ---- reinit params
        reinit_dt: float = 0.5,
        reinit_sign_eps: float = 0.02,
        reinit_narrow_band: float  = 5.0,
        reinit_target_grad_range: tuple = (0.98, 1.02),
        reinit_tol: float = 0.1,
        reinit_max_iters: int = 30,
        reinit_verbose: bool = False,
    ):
        super().__init__()
        self.es_D1 = float(es_D1)
        self.es_D2 = float(es_D2)
        self.es_D3 = float(es_D3)
        self.es_D4 = float(es_D4)
        self.ext_sigma = float(ext_sigma)
        self.max_bias_nm = float(max_bias_nm)
        self.phi_tau = float(phi_tau)
        self.render_tau = float(render_tau)
        self.edge_tau = float(edge_tau)
        self.edge_thr = float(edge_thr)
        self.truncate = float(truncate)
        self.band_r = int(band_r)
        self.edge_bin_thr = float(edge_bin_thr)

        # reinit config
        self.reinit_kwargs = {
            "dt": float(reinit_dt),
            "sign_eps": float(reinit_sign_eps),
            "narrow_band": None if reinit_narrow_band is None else float(reinit_narrow_band),
            "target_grad_range": tuple(reinit_target_grad_range),
            "tol": float(reinit_tol),
            "max_iters": int(reinit_max_iters),
            "verbose": bool(reinit_verbose),
        }

    def forward(self, mask_in: torch.Tensor):
        x = mask_in.float() if not torch.is_floating_point(mask_in) else mask_in
        x4 = to_4d_mask(x)
        m = clamp01(x4)

        w_edge = sobel_edge_weight(m, edge_tau=self.edge_tau, edge_thr=self.edge_thr)
        with torch.no_grad():
            edge_bin = (w_edge.detach() > self.edge_bin_thr)

        kD1, kD2, kD3, kD4, kExt = build_kernels(
            self.es_D1, self.es_D2, self.es_D3, self.es_D4,
            self.ext_sigma, self.truncate, device=m.device, dtype=m.dtype
        )

        D1, D2, D3, D4, raw = compute_features_and_raw(
            m, edge_bin, kD1, kD2, kD3, kD4, self.band_r,
        )

        # "bias_raw < 0" indicates inward contraction.
        bias_raw = torch.clamp(-raw, 0.0, self.max_bias_nm)

        bias_ext = extend_bias(bias_raw, w_edge, kExt)
        bias_ext = torch.clamp(bias_ext, 0.0, self.max_bias_nm)

        m_safe = torch.clamp(m, 1e-6, 1.0 - 1e-6)
        phi0 = self.phi_tau * torch.log((1.0 - m_safe) / m_safe)

        phi_sdf, band_mask = reinit_levelset_dynamic(phi0=phi0, **self.reinit_kwargs)

        phi_etch = phi_sdf + bias_ext
        mask_etch_soft = torch.sigmoid(-phi_etch / max(self.render_tau, 1e-6))

        return  mask_etch_soft, bias_ext

def demo_grad_check(
    image_path: str  = None,
    device: str = "cuda",
    H: int = 256,
    W: int = 256,
    seed: int = 0,
    resize_if_needed: bool = True,
):
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    if image_path is None:
        torch.manual_seed(seed)
        mask = torch.rand((H, W), device=dev, dtype=torch.float32)
        mask = torch.sigmoid((mask - 0.5) / 0.08)
    else:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"image_path not found: {image_path}")

        img = Image.open(image_path).convert("L")
        arr = (np.array(img) / 255.0).astype(np.float32)
        mask = torch.from_numpy(arr).to(dev)

        if resize_if_needed and (mask.shape[0] != H or mask.shape[1] != W):
            mask4 = mask[None, None, ...]
            mask4 = F.interpolate(mask4, size=(H, W), mode="bilinear", align_corners=False)
            mask = mask4[0, 0]

    mask = mask.clone().detach().requires_grad_(True)

    etch = DiffVEBEtch2D(
        es_D1=20.0, es_D2=50.0, es_D3=50.0, es_D4=100.0,
        ext_sigma=5.0,
        max_bias_nm=20.0,
        phi_tau=2.0,
        render_tau=0.02,
        edge_tau=0.02, edge_thr=0.03,
    ).to(dev)

    out,bias = etch(mask)

    m_in = to_4d_mask(mask)
    m_out = out
    loss = F.mse_loss(m_out, m_in)

    loss.backward()

    g = mask.grad
    if g is None:
        raise RuntimeError("mask.grad is None (gradient did not flow back to input mask)")

    has_nan = torch.isnan(g).any().item()
    has_inf = torch.isinf(g).any().item()
    nonzero = int((g.abs() > 0).sum().detach().cpu())

    print("loss =", float(loss.detach().cpu()))
    print(
        "grad stats: mean=%.6e std=%.6e max=%.6e min=%.6e nonzero=%d/%d nan=%s inf=%s"
        % (
            float(g.mean().detach().cpu()),
            float(g.std().detach().cpu()),
            float(g.max().detach().cpu()),
            float(g.min().detach().cpu()),
            nonzero,
            g.numel(),
            str(has_nan),
            str(has_inf),
        )
    )

    return out, mask, loss


if __name__ == "__main__":
    out, mask, loss = demo_grad_check(
        image_path="./samples/test_32nm.png",
        device="cuda",
        H=2048,
        W=2048,
    )