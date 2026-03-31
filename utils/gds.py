import numpy as np
import cv2
import gdstk


class Design:
    """
    A GDS Design wrapper that mimics utils/glp.py::Design interface:
      - polygons (list of [ [x,y], ... ])
      - center(sizeX, sizeY, offsetX, offsetY)
      - mat(sizeX, sizeY, offsetX, offsetY) -> float mask in [0,1]
    """

    def __init__(
        self,
        filename: str = None,
        down: int = 1,
        cell_name: str = None,
        layer: int = None,
        datatype: int = None,
        pixels_per_unit: float = 1000.0,
        flatten: bool = True,
    ):
        self._filename = filename
        self._polygons = []
        if filename is None:
            return

        lib = gdstk.read_gds(filename)

        # select cell
        if cell_name is None:
            tops = lib.top_level()
            if len(tops) == 0:
                raise ValueError(f"No top-level cell found in {filename}")
            cell = tops[0]
        else:
            cells = {c.name: c for c in lib.cells}
            if cell_name not in cells:
                raise ValueError(f"Cell '{cell_name}' not found in {filename}")
            cell = cells[cell_name]

        # expand references (recommended)
        if flatten:
            # gdstk versions differ: some require name argument for copy()
            try:
                cell = cell.copy(f"{cell.name}__flat")
            except TypeError:
                cell = cell.copy()
            cell.flatten()

        polys = []

        # collect polygons
        for poly in cell.polygons:
            if layer is not None and poly.layer != layer:
                continue
            if datatype is not None and poly.datatype != datatype:
                continue
            polys.append(poly.points)  # (N,2)

        # collect paths (optional): convert to polygons
        for path in getattr(cell, "paths", []):
            if layer is not None and getattr(path, "layer", None) != layer:
                continue
            if datatype is not None and getattr(path, "datatype", None) != datatype:
                continue
            for pts in path.to_polygons():
                polys.append(pts)

        if len(polys) == 0:
            raise ValueError(
                f"No polygons found in {filename} (cell={cell.name}, layer={layer}, datatype={datatype})"
            )

        scale = float(pixels_per_unit) / max(int(down), 1)

        for p in polys:
            pts = np.asarray(p, dtype=np.float64) * scale
            coords = [[int(round(x)), int(round(y))] for x, y in pts]
            if len(coords) >= 3:
                self._polygons.append(coords)

    @property
    def polygons(self):
        return self._polygons

    def range(self):
        minX, minY = 1e12, 1e12
        maxX, maxY = -1e12, -1e12
        for polygon in self._polygons:
            for x, y in polygon:
                if x < minX:
                    minX = x
                if y < minY:
                    minY = y
                if x > maxX:
                    maxX = x
                if y > maxY:
                    maxY = y
        return minX, minY, maxX, maxY

    def move(self, deltaX, deltaY):
        for polygon in self._polygons:
            for pt in polygon:
                pt[0] += int(deltaX)
                pt[1] += int(deltaY)

    def center(self, sizeX=2048, sizeY=2048, offsetX=512, offsetY=512):
        canvas = self.range()
        canvasX = canvas[2] - canvas[0]
        canvasY = canvas[3] - canvas[1]
        halfX = (sizeX - canvasX) // 2
        halfY = (sizeY - canvasY) // 2
        deltaX = halfX - canvas[0]
        deltaY = halfY - canvas[1]
        self.move(deltaX - offsetX, deltaY - offsetY)

    def image(self, sizeX=2048, sizeY=2048, offsetX=512, offsetY=512):
        polys = list(map(lambda x: np.array(x, np.int64) + np.array([[offsetX, offsetY]]), self._polygons))
        img = np.zeros([sizeX, sizeY], dtype=np.float32)
        for p in polys:
            img = cv2.fillPoly(img, [p], color=255)
        return img

    def mat(self, sizeX=2048, sizeY=2048, offsetX=512, offsetY=512):
        return self.image(sizeX, sizeY, offsetX, offsetY) / 255.0