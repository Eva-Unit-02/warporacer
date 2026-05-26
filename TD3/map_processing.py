from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from cv2 import IMREAD_GRAYSCALE, imread
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter
from scipy.spatial import KDTree
from skimage.morphology import skeletonize
from yaml import safe_load

try:
    from .config import NEIGHBOR_OFFSETS, OCCUPANCY_THRESHOLD, SMOOTHING_WINDOW
except ImportError:
    from config import NEIGHBOR_OFFSETS, OCCUPANCY_THRESHOLD, SMOOTHING_WINDOW


class Map:
    def __init__(self, yaml_path: Path):
        metadata = safe_load(yaml_path.read_text())
        image_path = yaml_path.parent / metadata["image"]
        grayscale = imread(str(image_path), IMREAD_GRAYSCALE)
        if grayscale is None:
            raise FileNotFoundError(image_path)

        self.meta = metadata
        self.raw = grayscale
        self.origin_x, self.origin_y, _ = metadata["origin"]
        self.resolution = float(metadata["resolution"])
        self.height, self.width = grayscale.shape

        driveable = grayscale >= OCCUPANCY_THRESHOLD
        self.distance_field = distance_transform_edt(driveable)
        self._extract_centerline(driveable)
        self._build_waypoint_lookup()

    @staticmethod
    def _adjacent_track_pixels(mask: np.ndarray, row: int, col: int) -> list[tuple[int, int]]:
        height, width = mask.shape
        out: list[tuple[int, int]] = []
        for d_row, d_col in NEIGHBOR_OFFSETS:
            next_row = row + d_row
            next_col = col + d_col
            if 0 <= next_row < height and 0 <= next_col < width and mask[next_row, next_col]:
                out.append((next_row, next_col))
        return out

    def _extract_centerline(self, driveable: np.ndarray):
        skeleton = skeletonize(driveable)
        skeleton_points = np.argwhere(skeleton)
        if skeleton_points.size == 0:
            raise RuntimeError("Unable to extract a centerline from the map image")

        seed_px = np.array(
            [self.height - 1 + self.origin_y / self.resolution, -self.origin_x / self.resolution]
        )
        seed_index = np.argmin(np.square(skeleton_points - seed_px).sum(axis=1))
        seed = tuple(int(v) for v in skeleton_points[seed_index])
        choices = self._adjacent_track_pixels(skeleton, seed[0], seed[1])
        if len(choices) < 2:
            raise RuntimeError(f"Skeleton seed {seed} only has {len(choices)} neighbors")

        start_node, goal_node = choices[0], choices[1]
        parent: dict[tuple[int, int], tuple[int, int]] = {start_node: start_node}
        frontier = deque([start_node])

        while frontier:
            row, col = frontier.popleft()
            for next_node in self._adjacent_track_pixels(skeleton, row, col):
                if next_node == seed or next_node in parent:
                    continue
                parent[next_node] = (row, col)
                if next_node == goal_node:
                    frontier.clear()
                    break
                frontier.append(next_node)

        trace = [seed]
        current = goal_node
        while current != start_node:
            trace.append(current)
            current = parent[current]
        trace.append(start_node)
        trace.reverse()

        rc = np.asarray(trace, dtype=np.float32)
        centerline_world = np.column_stack(
            (
                self.origin_x + rc[:, 1] * self.resolution,
                self.origin_y + (self.height - 1 - rc[:, 0]) * self.resolution,
            )
        )
        self.centerline = savgol_filter(
            centerline_world,
            SMOOTHING_WINDOW,
            3,
            axis=0,
            mode="wrap",
        )
        diffs = np.diff(self.centerline, axis=0, append=self.centerline[:1])
        self.heading = np.arctan2(diffs[:, 1], diffs[:, 0])
        average_spacing = float(np.linalg.norm(diffs, axis=1).mean())
        self.lookahead_stride = max(1, int(round(1.0 / average_spacing)))

    def _build_waypoint_lookup(self):
        centerline_px = np.column_stack(
            (
                self.height - 1 - (self.centerline[:, 1] - self.origin_y) / self.resolution,
                (self.centerline[:, 0] - self.origin_x) / self.resolution,
            )
        )
        kd_tree = KDTree(centerline_px)
        rows, cols = np.mgrid[: self.height, : self.width]
        query_points = np.column_stack((rows.ravel(), cols.ravel()))
        nearest = kd_tree.query(query_points, workers=-1)[1]
        self.closest_waypoint = nearest.reshape(rows.shape)
