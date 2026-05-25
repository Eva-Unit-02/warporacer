from collections import deque
from pathlib import Path

from cv2 import IMREAD_GRAYSCALE, imread
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter
from scipy.spatial import KDTree
from skimage.morphology import skeletonize
from yaml import safe_load

try:
    from .config import ADJ, OCC_THRESH, SMOOTH_WINDOW
except ImportError:
    from config import ADJ, OCC_THRESH, SMOOTH_WINDOW


class Map:
    def __init__(self, path: Path):
        self.meta = safe_load(path.read_text())
        img_path = path.parent / self.meta["image"]
        self.raw = imread(str(img_path), IMREAD_GRAYSCALE)
        if self.raw is None:
            raise FileNotFoundError(img_path)
        free = self.raw >= OCC_THRESH
        self.dt = distance_transform_edt(free)
        self.ox, self.oy, _ = self.meta["origin"]
        self.h, self.w = self.raw.shape
        self.res = float(self.meta["resolution"])
        self._compute_centerline(free)
        self._build_lut()

    @staticmethod
    def _neighbors(skel, row, col, height, width):
        return [
            (row + d_row, col + d_col)
            for d_row, d_col in ADJ
            if 0 <= row + d_row < height
            and 0 <= col + d_col < width
            and skel[row + d_row, col + d_col]
        ]

    def _compute_centerline(self, free):
        skel = skeletonize(free)
        height, width = skel.shape
        points = np.argwhere(skel)
        origin_px = np.array([self.h - 1 + self.oy / self.res, -self.ox / self.res])
        start = tuple(int(x) for x in points[np.argmin(((points - origin_px) ** 2).sum(1))])
        neighbors = self._neighbors(skel, start[0], start[1], height, width)
        if len(neighbors) < 2:
            raise RuntimeError(f"Skeleton seed {start} has {len(neighbors)} neighbours")
        src, target = neighbors[0], neighbors[1]
        parent = {src: src}
        queue = deque([src])
        while queue:
            row, col = queue.popleft()
            for next_row, next_col in self._neighbors(skel, row, col, height, width):
                neighbor = (next_row, next_col)
                if neighbor in parent or neighbor == start:
                    continue
                parent[neighbor] = (row, col)
                if neighbor == target:
                    queue.clear()
                    break
                queue.append(neighbor)
        path = [start]
        node = target
        while node != src:
            path.append(node)
            node = parent[node]
        path.append(src)
        path.reverse()
        rc = np.array(path)
        world = np.column_stack(
            [
                self.ox + rc[:, 1] * self.res,
                self.oy + (self.h - 1 - rc[:, 0]) * self.res,
            ]
        )
        self.centerline = savgol_filter(world, SMOOTH_WINDOW, 3, axis=0, mode="wrap")
        diffs = np.diff(self.centerline, axis=0, append=self.centerline[:1])
        self.angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        avg_spacing = float(np.linalg.norm(diffs, axis=1).mean())
        self.look_step = max(1, int(round(1.0 / avg_spacing)))

    def _build_lut(self):
        cl_px = np.column_stack(
            [
                self.h - 1 - (self.centerline[:, 1] - self.oy) / self.res,
                (self.centerline[:, 0] - self.ox) / self.res,
            ]
        )
        tree = KDTree(cl_px)
        rows, cols = np.mgrid[: self.h, : self.w]
        self.lut = tree.query(
            np.column_stack([rows.ravel(), cols.ravel()]), workers=-1
        )[1].reshape(rows.shape)
