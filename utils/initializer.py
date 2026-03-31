import sys

sys.path.append(".")

import numpy as np

import torch


REALTYPE = torch.float32
COMPLEXTYPE = torch.complex64
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Initializer:
    def __init__(self):
        pass

    def run(self, design, sizeX, sizeY, offsetX, offsetY, dtype=REALTYPE, device=DEVICE):
        pass

def _distMatPolygon(polygon, canvas, offsets):
    if len(canvas) == 4:
        canvas = [[canvas[0], canvas[1]], [canvas[2], canvas[3]]]
    minX, minY, maxX, maxY = canvas[0][0], canvas[0][1], canvas[1][0], canvas[1][1]
    sizeX, sizeY = maxX - minX, maxY - minY

    dist = np.ones([sizeX, sizeY]) * (sizeX * sizeY)
    xs = np.arange(minX, maxX, 1, dtype=np.int32).reshape([sizeX, 1])
    ys = np.arange(minY, maxY, 1, dtype=np.int32).reshape([1, sizeY])
    xs = np.tile(xs, [1, sizeY])
    ys = np.tile(ys, [sizeX, 1])

    frPt = polygon[-1]
    for toPt in polygon:
        frX, frY = frPt
        toX, toY = toPt
        if frX > toX:
            frX, toX = toX, frX
        if frY > toY:
            frY, toY = toY, frY
        frX += offsets[0]
        toX += offsets[0]
        frY += offsets[1]
        toY += offsets[1]

        dist1 = np.sqrt((frX - xs) ** 2 + (frY - ys) ** 2)
        dist2 = np.sqrt((toX - xs) ** 2 + (toY - ys) ** 2)

        dist = np.minimum(dist, np.minimum(dist1, dist2))

        if frX == toX:
            mask = (frY <= ys) * (ys <= toY)
            new = np.minimum(dist, np.abs(frX - xs))
            dist[mask] = new[mask]
        elif frY == toY:
            mask = (frX <= xs) * (xs <= toX)
            new = np.minimum(dist, np.abs(frY - ys))
            dist[mask] = new[mask]

        frPt = toPt
    return dist.T


def _distMatLegacy(design, canvas=[[0, 0], [2048, 2048]], offsets=[512, 512]):
    if len(canvas) == 4:
        canvas = [[canvas[0], canvas[1]], [canvas[2], canvas[3]]]
    minX, minY, maxX, maxY = canvas[0][0], canvas[0][1], canvas[1][0], canvas[1][1]

    mask = design.mat(sizeX=maxX - minX, sizeY=maxY - minY, offsetX=offsets[0], offsetY=offsets[1])
    dist = np.ones([maxX - minX, maxY - minY]) * ((maxX - minX) * (maxY - minY))
    for polygon in design.polygons:
        tmp = _distMatPolygon(polygon, canvas, offsets)
        dist = np.minimum(dist, tmp)
    dist[mask > 0] *= -1
    return dist


# torch版本_distMatPolygon
def _distMatPolygonTorch(polygon, canvas, offsets):
    if len(canvas) == 4:
        canvas = [[canvas[0], canvas[1]], [canvas[2], canvas[3]]]
    minX, minY, maxX, maxY = canvas[0][0], canvas[0][1], canvas[1][0], canvas[1][1]
    sizeX, sizeY = maxX - minX, maxY - minY

    dist = torch.ones([sizeX, sizeY], dtype=REALTYPE, device=DEVICE) * (sizeX * sizeY)

    xs = np.arange(minX, maxX, 1, dtype=np.int32).reshape([sizeX, 1])
    ys = np.arange(minY, maxY, 1, dtype=np.int32).reshape([1, sizeY])
    xs = torch.tensor(np.tile(xs, [1, sizeY]), dtype=REALTYPE, device=DEVICE)
    ys = torch.tensor(np.tile(ys, [sizeX, 1]), dtype=REALTYPE, device=DEVICE)

    frPt = polygon[-1]
    for toPt in polygon:
        frX, frY = frPt
        toX, toY = toPt

        if frX > toX: frX, toX = toX, frX
        if frY > toY: frY, toY = toY, frY

        frX += offsets[0]
        toX += offsets[0]
        frY += offsets[1]
        toY += offsets[1]

        dist1 = torch.sqrt((frX - xs) ** 2 + (frY - ys) ** 2)
        dist2 = torch.sqrt((toX - xs) ** 2 + (toY - ys) ** 2)

        dist = torch.minimum(dist, torch.minimum(dist1, dist2))

        if frX == toX:
            mask = (frY <= ys) * (ys <= toY)
            new_dist = torch.abs(frX - xs)
            dist[mask] = torch.minimum(dist[mask], new_dist[mask])

        elif frY == toY:
            mask = (frX <= xs) * (xs <= toX)
            new_dist = torch.abs(frY - ys)
            dist[mask] = torch.minimum(dist[mask], new_dist[mask])

        frPt = toPt

    return dist.T


# torch版本_distMatLegacy
def _distMatTorch(design, canvas=[[0, 0], [2048, 2048]], offsets=[512, 512], mask=None):
    if len(canvas) == 4:
        canvas = [[canvas[0], canvas[1]], [canvas[2], canvas[3]]]
    minX, minY, maxX, maxY = canvas[0][0], canvas[0][1], canvas[1][0], canvas[1][1]

    if mask is None:
        mask = design.mat(sizeX=maxX - minX, sizeY=maxY - minY, offsetX=offsets[0], offsetY=offsets[1])
    dist = torch.ones([maxX - minX, maxY - minY], dtype=REALTYPE, device=DEVICE) * ((maxX - minX) * (maxY - minY))
    for polygon in design.polygons:
        tmp = _distMatPolygonTorch(polygon, canvas, offsets)
        dist = torch.minimum(dist, tmp)
    dist[mask > 0] *= -1
    return dist


class LevelSetInitTorch(Initializer):
    def __init__(self):
        super(LevelSetInitTorch, self).__init__()

    def run(self, design, sizeX, sizeY, offsetX, offsetY, dtype=REALTYPE, device=DEVICE):
        target = torch.tensor(design.mat(sizeX, sizeY, offsetX, offsetY), dtype=dtype, device=device)
        params = _distMatTorch(design, canvas=[[0, 0], [sizeX, sizeY]],
                               offsets=[offsetX, offsetY]).detach().clone().requires_grad_(True)
        return target, params




