import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pathlib import Path
import time

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func

import utils.gds as gds

import utils.initializer as initializer
import src.etch_sim as etch_sim
import argparse


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REALTYPE = torch.float32

ITERATIONS = 10
STEP_SIZE = 0.8

def gradImage(image):
    GRAD_STEPSIZE = 1.0
    image = image.view([-1, 1, image.shape[-2], image.shape[-1]])
    padded = func.pad(image, (1, 1, 1, 1), mode="replicate")[:, 0].detach()
    gradX = (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]) / (2.0 * GRAD_STEPSIZE)
    gradY = (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]) / (2.0 * GRAD_STEPSIZE)
    return gradX.view(image.shape), gradY.view(image.shape)


class _Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, levelset):
        ctx.save_for_backward(levelset)
        mask = torch.zeros_like(levelset)
        mask[levelset < 0] = 1.0
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        (levelset,) = ctx.saved_tensors
        gradX, gradY = gradImage(levelset)
        l2norm = torch.sqrt(gradX ** 2 + gradY ** 2 + 1e-12)
        return -l2norm * grad_output


class Binarize(nn.Module):
    def forward(self, levelset):
        return _Binarize.apply(levelset)


class LevelSet(nn.Module):
    def __init__(self, device):
        super(LevelSet, self).__init__()
        self._binarize = Binarize()

        self.etch = etch_sim.DiffVEBEtch2D(
            es_D1=20.0, es_D2=25.0, es_D3=30.0, es_D4=100.0,
            ext_sigma=5.0,
            max_bias_nm=20.0,
            phi_tau=5.0,
            render_tau=0.02,
            edge_tau=0.01, edge_thr=0.1, truncate=1
        ).to(device)

    def forward(self, params):
        retarget = self._binarize(params)

        mask_etch_soft, bias_ext = self.etch(retarget)

        return retarget, bias_ext.squeeze().squeeze()


class LevelSetILT:
    def __init__(self, device):
        self.device = torch.device(device)

        self._levelset = LevelSet(self.device).to(self.device)
        self._binarize = Binarize()

        self._filter = torch.zeros([2048, 2048], dtype=REALTYPE, device=self.device)
        self._filter[512:512 + 1024, 512:512 + 1024] = 1.0

    def solve(self, target, params, verbose: int = 0):
        backup = params
        params = params.clone().detach().requires_grad_(True)

        opt = optim.Adam([params], lr=STEP_SIZE)

        best_loss = 1e30
        best_params = None
        best_retarget = None
        best_etch = None
        best_bias = None

        for it in range(ITERATIONS):
            retarget, ext_bias = self._levelset(params * self._filter + backup * (1.0 - self._filter))
            l2loss = func.mse_loss(self._binarize(params + ext_bias), target, reduction="sum")
            loss = l2loss

            if verbose:
                print(f"[Iter {it:04d}] L2={l2loss.item():.0f}")

            if l2loss.item() < best_loss and it < 11 :
                best_loss = l2loss.item()
                best_params = params.detach().clone()
                best_retarget = retarget.detach().clone()
                best_bias = ext_bias.detach().clone()
                best_etch = self._binarize(best_params + best_bias).detach().clone()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        return best_loss, best_params, best_retarget, best_etch, best_bias


def save_u8(path: Path, x: torch.Tensor):
    arr = (x.detach().cpu().numpy() * 255.0).astype(np.uint8)
    cv2.imwrite(str(path), arr)


def run_epc(gds_path: str, out_dir: str, verbose: int = 0):
    dev = torch.device(DEVICE)

    gds_path = Path(gds_path)
    if not gds_path.exists():
        raise FileNotFoundError(f"Missing gds file: {gds_path}")
    if gds_path.suffix.lower() != ".gds":
        raise ValueError(f"Input must be a .gds file, got: {gds_path}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


    stem = gds_path.stem

    solver = LevelSetILT(DEVICE)

    design = gds.Design(str(gds_path), down=1)
    design.center()

    target, params = initializer.LevelSetInitTorch().run(design, 2048, 2048, 512, 512)
    target = target.to(dev)
    params = params.to(dev)

    t0 = time.time()
    l2, best_params, best_retarget, best_etch, best_bias = solver.solve(target, params, verbose=verbose)
    dt = time.time() - t0

    best_retarget = (best_retarget > 0.5).float()
    best_etch = (best_etch > 0.5).float()

    save_u8(out_dir / f"{stem}_target.png", target)
    save_u8(out_dir / f"{stem}_retarget.png", best_retarget)
    save_u8(out_dir / f"{stem}_etch.png", best_etch)

    if verbose:
        print(f"[OK] {gds_path.name} | L2={l2:.0f} runtime={dt:.2f}s -> {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gds", required=True, help="path to a single .gds file")
    p.add_argument("--out", required=True, help="output folder for png results")
    p.add_argument("--verbose", type=int, default=0)
    args = p.parse_args()

    run_epc(gds_path=args.gds, out_dir=args.out, verbose=args.verbose)
