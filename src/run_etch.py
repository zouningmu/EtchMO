import os
import argparse
import numpy as np
import cv2
from PIL import Image

import torch

from etch_sim import DiffVEBEtch2D


@torch.no_grad()
def run_etch_only(
    image_path: str,
    out_dir: str,
    device: str = "cuda",
    H: int = 2048,
    W: int = 2048,
):
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    img = Image.open(image_path).convert("L")
    if (H is not None) and (W is not None) and img.size != (W, H):
        img = img.resize((W, H), resample=Image.BILINEAR)

    arr = (np.array(img) / 255.0).astype(np.float32)

    # ===== NEW: binarize input (threshold=0.225) and save =====
    bin_arr = (arr > 0.5).astype(np.uint8) * 255
    bin_img = Image.fromarray(bin_arr)
    bin_save_path = os.path.join(out_dir, "printed.png")
    bin_img.save(bin_save_path)

    # use soft input  for etch
    mask = torch.from_numpy(arr).to(dev)

    etch = DiffVEBEtch2D(
        es_D1=20.0, es_D2=50.0, es_D3=50.0, es_D4=100.0,
        ext_sigma=5.0,
        max_bias_nm=20.0,
        phi_tau=2.0,
        render_tau=0.02,
        edge_tau=0.02,
        edge_thr=0.03,
    ).to(dev)

    out, bias = etch(mask)

    etch_img = out.squeeze().detach().cpu().numpy()
    etch_img = (etch_img * 255).astype(np.uint8)

    etch_img[etch_img > 127] = 255
    etch_img[etch_img <= 127] = 0
    kernel = np.ones((3, 3), np.uint8)
    etch_img = cv2.morphologyEx(etch_img, cv2.MORPH_OPEN, kernel)

    etch_img = Image.fromarray(etch_img)
    save_path = os.path.join(out_dir, "etch.png")
    etch_img.save(save_path)

    print(f"[OK] saved: {bin_save_path}")
    print(f"[OK] saved: {save_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--H", type=int, default=None)
    p.add_argument("--W", type=int, default=None)
    args = p.parse_args()

    run_etch_only(
        image_path=args.input,
        out_dir=args.out,
        device=args.device,
        H=args.H,
        W=args.W,
    )