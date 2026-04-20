import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import warp as wp
from cv2 import (
    COLOR_GRAY2RGB,
    IMREAD_GRAYSCALE,
    cvtColor,
    fillPoly,
    imread,
    polylines,
)
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter
from scipy.spatial import KDTree
from skimage.morphology import skeletonize
from torch.distributions import Normal
from typer import run
from yaml import safe_load

OCC_THRESH = 230
SMOOTH_WINDOW = 51
LIDAR_RANGE = 20.0
NUM_LIDAR = 108
LIDAR_FOV = np.radians(270.0)
OBS_DIM = 3 + NUM_LIDAR
ACT_DIM = 2
MAX_STEPS = 10000
ADJ = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

MU = 1.0489
FRONT_CORNERING_STIFFNESS = 4.718
REAR_CORNERING_STIFFNESS = 5.4562
LENGTH_FRONT = 0.15875
LENGTH_REAR = 0.17145
LENGTH_WHEELBASE = LENGTH_FRONT + LENGTH_REAR
CG_HEIGHT = 0.074
MASS = 3.74
INERTIA = 0.04712
STEER_MIN = -0.4189
STEER_MAX = 0.4189
STEER_V_MAX = 3.2
V_SWITCH = 7.319
A_MAX = 9.51
V_MIN = -5.0
V_MAX = 20.0
PSI_PRIME_MAX = 6.0
WIDTH = 0.31
LENGTH = 0.58
G = 9.81
DT = 1 / 60

DONE_TERMINATED = 1
DONE_TRUNCATED = 2

OBS_LOW = np.concatenate(
    [[STEER_MIN, V_MIN, -PSI_PRIME_MAX], np.zeros(NUM_LIDAR)]
).astype(np.float32)
OBS_HIGH = np.concatenate(
    [[STEER_MAX, V_MAX, PSI_PRIME_MAX], np.full(NUM_LIDAR, LIDAR_RANGE)]
).astype(np.float32)


@wp.struct
class VehicleD:
    d_x: float
    d_y: float
    d_delta: float
    d_v: float
    d_psi: float
    d_beta: float
    dd_psi: float


@wp.func
def vehicle_dynamics_st(
    car_delta: float,
    car_v: float,
    car_psi: float,
    car_psi_prime: float,
    car_beta: float,
    steer_v: float,
    acceleration: float,
):
    state = VehicleD()
    state.d_delta = steer_v
    state.d_v = acceleration
    if wp.abs(car_v) < V_SWITCH:
        beta = wp.atan(wp.tan(car_delta) * LENGTH_REAR / LENGTH_WHEELBASE)
        state.d_x = car_v * wp.cos(beta + car_psi)
        state.d_y = car_v * wp.sin(beta + car_psi)
        state.d_psi = car_v * wp.cos(beta) * wp.tan(car_delta) / LENGTH_WHEELBASE
        state.d_beta = (LENGTH_REAR * steer_v) / (
            LENGTH_WHEELBASE
            * wp.cos(car_delta) ** 2.0
            * (1.0 + (wp.tan(car_delta) * LENGTH_REAR / LENGTH_WHEELBASE) ** 2.0)
        )
        state.dd_psi = (
            1.0
            / LENGTH_WHEELBASE
            * (
                acceleration * wp.cos(car_beta) * wp.tan(car_delta)
                - car_v * wp.sin(car_beta) * state.d_beta * wp.tan(car_delta)
                + car_v * wp.cos(car_beta) * steer_v / wp.cos(car_delta) ** 2.0
            )
        )
    else:
        state.d_x = car_v * wp.cos(car_beta + car_psi)
        state.d_y = car_v * wp.sin(car_beta + car_psi)
        state.d_psi = car_psi_prime
        state.dd_psi = (
            -MU
            * MASS
            / (car_v * INERTIA * LENGTH_WHEELBASE)
            * (
                LENGTH_FRONT**2.0
                * FRONT_CORNERING_STIFFNESS
                * (G * LENGTH_REAR - acceleration * CG_HEIGHT)
                + LENGTH_REAR**2.0
                * REAR_CORNERING_STIFFNESS
                * (G * LENGTH_FRONT + acceleration * CG_HEIGHT)
            )
            * car_psi_prime
            + MU
            * MASS
            / (INERTIA * LENGTH_WHEELBASE)
            * (
                LENGTH_REAR
                * REAR_CORNERING_STIFFNESS
                * (G * LENGTH_FRONT + acceleration * CG_HEIGHT)
                - LENGTH_FRONT
                * FRONT_CORNERING_STIFFNESS
                * (G * LENGTH_REAR - acceleration * CG_HEIGHT)
            )
            * car_beta
            + MU
            * MASS
            / (INERTIA * LENGTH_WHEELBASE)
            * LENGTH_FRONT
            * FRONT_CORNERING_STIFFNESS
            * (G * LENGTH_REAR - acceleration * CG_HEIGHT)
            * car_delta
        )
        state.d_beta = (
            (
                MU
                / (car_v**2.0 * LENGTH_WHEELBASE)
                * (
                    REAR_CORNERING_STIFFNESS
                    * (G * LENGTH_FRONT + acceleration * CG_HEIGHT)
                    * LENGTH_REAR
                    - FRONT_CORNERING_STIFFNESS
                    * (G * LENGTH_REAR - acceleration * CG_HEIGHT)
                    * LENGTH_FRONT
                )
                - 1.0
            )
            * car_psi_prime
            - MU
            / (car_v * LENGTH_WHEELBASE)
            * (
                REAR_CORNERING_STIFFNESS * (G * LENGTH_FRONT + acceleration * CG_HEIGHT)
                + FRONT_CORNERING_STIFFNESS
                * (G * LENGTH_REAR - acceleration * CG_HEIGHT)
            )
            * car_beta
            + MU
            / (car_v * LENGTH_WHEELBASE)
            * (FRONT_CORNERING_STIFFNESS * (G * LENGTH_REAR - acceleration * CG_HEIGHT))
            * car_delta
        )
    return state


@wp.kernel
def step_kernel(
    actions: wp.array[wp.vec2],
    observation: wp.array2d[float],
    reward: wp.array[float],
    done: wp.array[int],
    cars: wp.array2d[float],
    cars_int: wp.array2d[int],
    origin: wp.vec2,
    res: float,
    distance_transform_px: wp.array2d[float],
    centerline_lut: wp.array2d[int],
    centerline: wp.array[wp.vec3],
    num_centerline_pts: int,
    lidar_dirs: wp.array[wp.vec2],
):
    i = wp.tid()

    car_x = cars[i, 0]
    car_y = cars[i, 1]
    car_delta = cars[i, 2]
    car_v = cars[i, 3]
    car_psi = cars[i, 4]
    car_psi_prime = cars[i, 5]
    car_beta = cars[i, 6]
    car_steps = cars_int[i, 0]
    car_waypoint = cars_int[i, 1]

    origin_x = origin[0]
    origin_y = origin[1]
    map_w = distance_transform_px.shape[0]
    map_h = distance_transform_px.shape[1]

    steer_action = actions[i][0]
    acceleration_action = actions[i][1]

    steer_v = wp.clamp(steer_action, -1.0, 1.0) * STEER_V_MAX
    if (steer_v < 0 and car_delta <= STEER_MIN) or (
        steer_v > 0 and car_delta >= STEER_MAX
    ):
        steer_v = 0.0
    acceleration = wp.clamp(acceleration_action, -1.0, 1.0) * A_MAX
    if (acceleration < 0 and car_v <= V_MIN) or (acceleration > 0 and car_v >= V_MAX):
        acceleration = 0.0

    k1 = vehicle_dynamics_st(
        car_delta, car_v, car_psi, car_psi_prime, car_beta, steer_v, acceleration
    )
    k2 = vehicle_dynamics_st(
        car_delta + k1.d_delta * DT * 0.5,
        car_v + k1.d_v * DT * 0.5,
        car_psi + k1.d_psi * DT * 0.5,
        car_psi_prime + k1.dd_psi * DT * 0.5,
        car_beta + k1.d_beta * DT * 0.5,
        steer_v,
        acceleration,
    )
    k3 = vehicle_dynamics_st(
        car_delta + k2.d_delta * DT * 0.5,
        car_v + k2.d_v * DT * 0.5,
        car_psi + k2.d_psi * DT * 0.5,
        car_psi_prime + k2.dd_psi * DT * 0.5,
        car_beta + k2.d_beta * DT * 0.5,
        steer_v,
        acceleration,
    )
    k4 = vehicle_dynamics_st(
        car_delta + k3.d_delta * DT,
        car_v + k3.d_v * DT,
        car_psi + k3.d_psi * DT,
        car_psi_prime + k3.dd_psi * DT,
        car_beta + k3.d_beta * DT,
        steer_v,
        acceleration,
    )

    car_x += (k1.d_x + 2.0 * k2.d_x + 2.0 * k3.d_x + k4.d_x) * DT / 6.0
    car_y += (k1.d_y + 2.0 * k2.d_y + 2.0 * k3.d_y + k4.d_y) * DT / 6.0
    car_delta += (
        (k1.d_delta + 2.0 * k2.d_delta + 2.0 * k3.d_delta + k4.d_delta) * DT / 6.0
    )
    car_v += (k1.d_v + 2.0 * k2.d_v + 2.0 * k3.d_v + k4.d_v) * DT / 6.0
    car_psi += (k1.d_psi + 2.0 * k2.d_psi + 2.0 * k3.d_psi + k4.d_psi) * DT / 6.0
    car_psi_prime += (
        (k1.dd_psi + 2.0 * k2.dd_psi + 2.0 * k3.dd_psi + k4.dd_psi) * DT / 6.0
    )
    car_beta += (k1.d_beta + 2.0 * k2.d_beta + 2.0 * k3.d_beta + k4.d_beta) * DT / 6.0

    car_delta = wp.clamp(car_delta, STEER_MIN, STEER_MAX)
    car_v = wp.clamp(car_v, V_MIN, V_MAX)
    car_psi_prime = wp.clamp(car_psi_prime, -PSI_PRIME_MAX, PSI_PRIME_MAX)
    car_beta = wp.clamp(car_beta, -1.2, 1.2)

    car_px = wp.clamp(wp.int32((car_x - origin_x) / res), 0, map_w - 1)
    car_py = wp.clamp(
        wp.int32(float(map_h) - 1.0 - (car_y - origin_y) / res), 0, map_h - 1
    )
    car_pos_px = wp.vec2(wp.float32(car_px), wp.float32(car_py))

    term = distance_transform_px[car_px, car_py] * res < wp.length(
        wp.vec2(WIDTH / 2.0, LENGTH / 2.0)
    )
    trunc = car_steps >= MAX_STEPS
    car_steps += 1

    new_car_waypoint = centerline_lut[car_px, car_py]
    d_centerline_pt = new_car_waypoint - car_waypoint
    if d_centerline_pt > num_centerline_pts / 2:
        d_centerline_pt -= num_centerline_pts
    elif d_centerline_pt < -num_centerline_pts / 2:
        d_centerline_pt += num_centerline_pts
    reward[i] = wp.float32(d_centerline_pt) / wp.float32(
        num_centerline_pts
    ) - wp.float32(term)

    if term:
        done[i] = 1
    elif trunc:
        done[i] = 2
    else:
        done[i] = 0

    if trunc or term:
        seed = i * 2654435761 + new_car_waypoint * 2246822519 + car_steps * 3266489917
        random_number = wp.int32(wp.uint32(seed) >> wp.uint32(16)) % num_centerline_pts
        car_x = centerline[random_number][0]
        car_y = centerline[random_number][1]
        car_delta = 0.0
        car_v = 0.0
        car_psi = centerline[random_number][2]
        car_psi_prime = 0.0
        car_beta = 0.0
        car_steps = 0
        new_car_waypoint = random_number
        car_px = wp.clamp(wp.int32((car_x - origin_x) / res), 0, map_w - 1)
        car_py = wp.clamp(
            wp.int32(float(map_h) - 1.0 - (car_y - origin_y) / res), 0, map_h - 1
        )
        car_pos_px = wp.vec2(wp.float32(car_px), wp.float32(car_py))

    sh, ch = wp.sin(car_psi), wp.cos(car_psi)
    for j in range(lidar_dirs.shape[0]):
        ray = wp.vec2(wp.float32(car_px), wp.float32(car_py))
        ca = lidar_dirs[j][0]
        sa = lidar_dirs[j][1]
        d_px = wp.vec2(ch * ca - sh * sa, -(sh * ca + ch * sa))
        while wp.length(ray - car_pos_px) * res < LIDAR_RANGE:
            ray_px = wp.int32(ray[0])
            ray_py = wp.int32(ray[1])
            if ray_px < 0 or ray_px >= map_w or ray_py < 0 or ray_py >= map_h:
                break
            dt_ray = distance_transform_px[ray_px, ray_py]
            ray += d_px * dt_ray
            if dt_ray == 0.0:
                break
        observation[i, j + 3] = wp.min(wp.length(ray - car_pos_px) * res, LIDAR_RANGE)

    cars[i, 0] = car_x
    cars[i, 1] = car_y
    cars[i, 2] = car_delta
    cars[i, 3] = car_v
    cars[i, 4] = car_psi
    cars[i, 5] = car_psi_prime
    cars[i, 6] = car_beta
    cars_int[i, 0] = car_steps
    cars_int[i, 1] = new_car_waypoint

    observation[i, 0] = car_delta
    observation[i, 1] = car_v
    observation[i, 2] = car_psi_prime


class Map:
    def __init__(self, path: Path) -> None:
        with open(path, "r") as f:
            self.meta = safe_load(f)
        self.raw = imread(str(path.parent / self.meta["image"]), IMREAD_GRAYSCALE)
        if self.raw is None:
            raise FileNotFoundError(path.parent / self.meta["image"])
        self.dt = distance_transform_edt(self.raw >= OCC_THRESH)
        self.ox, self.oy, self.ophi = self.meta["origin"]
        self.h, self.w = self.raw.shape
        self.res = float(self.meta["resolution"])
        self._compute_centerline(self.raw)
        self._build_lut()

    @staticmethod
    def _neighbors(skeleton, r, c):
        h, w = skeleton.shape
        return [
            (r + dr, c + dc)
            for dr, dc in ADJ
            if 0 <= r + dr < h and 0 <= c + dc < w and skeleton[r + dr, c + dc]
        ]

    def _compute_centerline(self, raw, smooth_window=SMOOTH_WINDOW):
        skeleton = skeletonize(raw >= OCC_THRESH)
        pts = np.argwhere(skeleton)
        origin_px = [self.h - 1 + self.oy / self.res, -self.ox / self.res]
        start = tuple(
            int(x) for x in pts[np.argmin(np.linalg.norm(pts - origin_px, axis=1))]
        )
        nbrs = self._neighbors(skeleton, *start)
        if len(nbrs) < 2:
            raise RuntimeError(
                f"Skeleton seed {start} has {len(nbrs)} neighbors; need a closed loop."
            )
        src, target = nbrs[0], nbrs[1]

        parent = {src: src}
        q = deque([src])
        while q:
            r, c = q.popleft()
            for nr, nc in self._neighbors(skeleton, r, c):
                n = (nr, nc)
                if n not in parent and n != start:
                    parent[n] = (r, c)
                    if n == target:
                        q.clear()
                        break
                    q.append(n)
        path = [start]
        n = target
        while n != src:
            path.append(n)
            n = parent[n]
        path.append(src)
        path.reverse()
        rc = np.array(path)
        world = np.column_stack(
            [
                self.ox + rc[:, 1] * self.res,
                self.oy + (self.h - 1 - rc[:, 0]) * self.res,
            ]
        )
        self.centerline = savgol_filter(world, smooth_window, 3, axis=0, mode="wrap")
        diffs = np.diff(self.centerline, axis=0, append=self.centerline[:1])
        self.angles = np.arctan2(diffs[:, 1], diffs[:, 0])

    def _build_lut(self):
        centerline_px = np.column_stack(
            [
                self.h - 1 - (self.centerline[:, 1] - self.oy) / self.res,
                (self.centerline[:, 0] - self.ox) / self.res,
            ]
        )
        kdtree = KDTree(centerline_px)
        rows, cols = np.mgrid[: self.h, : self.w]
        self.centerline_lut = kdtree.query(
            np.column_stack([rows.ravel(), cols.ravel()]), workers=-1
        )[1].reshape(rows.shape)


class RacingEnv:
    observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)

    def __init__(
        self,
        map_path: Path,
        num_envs: int,
        seed: int = 0,
        device: str | None = None,
    ):
        wp.init()
        self.num_envs = num_envs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.map = Map(map_path)
        d = self.device

        self.dt_buf = wp.array(self.map.dt.T.astype(np.float32), dtype=float, device=d)
        self.lut_buf = wp.array(
            self.map.centerline_lut.T.astype(np.int32), dtype=int, device=d
        )
        self.centerline_buf = wp.array(
            np.column_stack([self.map.centerline, self.map.angles]).astype(np.float32),
            dtype=wp.vec3,
            device=d,
        )
        self.num_centerline = len(self.map.centerline)

        rng = np.random.default_rng(seed)
        idxs = rng.integers(0, self.num_centerline, size=num_envs)
        cars = np.zeros((num_envs, 7), dtype=np.float32)
        cars[:, 0] = self.map.centerline[idxs, 0]
        cars[:, 1] = self.map.centerline[idxs, 1]
        cars[:, 4] = self.map.angles[idxs]
        cars_int = np.zeros((num_envs, 2), dtype=np.int32)
        cars_int[:, 1] = idxs

        self.cars = wp.array(cars, dtype=float, device=d)
        self.cars_int = wp.array(cars_int, dtype=int, device=d)
        self.obs = wp.zeros((num_envs, OBS_DIM), dtype=float, device=d)
        self.rew = wp.zeros(num_envs, dtype=float, device=d)
        self.done = wp.zeros(num_envs, dtype=int, device=d)

        self.obs_buf = wp.to_torch(self.obs)
        self.rew_buf = wp.to_torch(self.rew)
        self.done_buf = wp.to_torch(self.done)
        self._cars_buf = wp.to_torch(self.cars)
        self._step_counter = wp.to_torch(self.cars_int)[:, 0]

        angles = np.linspace(-LIDAR_FOV / 2, LIDAR_FOV / 2, NUM_LIDAR, dtype=np.float32)
        self.lidar_buf = wp.array(
            np.column_stack([np.cos(angles), np.sin(angles)]),
            dtype=wp.vec2,
            device=d,
        )

        self.nan_events = 0
        self._launch(wp.zeros(num_envs, dtype=wp.vec2, device=d))
        self._sanitize()

    def _launch(self, act):
        wp.launch(
            step_kernel,
            dim=self.num_envs,
            inputs=[
                act,
                self.obs,
                self.rew,
                self.done,
                self.cars,
                self.cars_int,
                wp.vec2(self.map.ox, self.map.oy),
                self.map.res,
                self.dt_buf,
                self.lut_buf,
                self.centerline_buf,
                self.num_centerline,
                self.lidar_buf,
            ],
        )
        wp.synchronize()

    def _sanitize(self):
        bad = (
            torch.isnan(self.obs_buf).any(dim=1)
            | torch.isinf(self.obs_buf).any(dim=1)
            | torch.isnan(self._cars_buf).any(dim=1)
            | torch.isinf(self._cars_buf).any(dim=1)
        )
        torch.nan_to_num_(self.obs_buf, nan=0.0, posinf=LIDAR_RANGE, neginf=0.0)
        torch.nan_to_num_(self._cars_buf, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nan_to_num_(self.rew_buf, nan=0.0, posinf=0.0, neginf=0.0)
        if bad.any():
            self.nan_events += int(bad.sum().item())
            self._step_counter[bad] = MAX_STEPS
            self.done_buf[bad] = DONE_TRUNCATED

    def reset(self):
        return self.obs_buf, {}

    def step(self, actions):
        self._launch(wp.from_torch(actions.detach().contiguous(), dtype=wp.vec2))
        self._sanitize()
        terminated = self.done_buf == DONE_TERMINATED
        truncated = self.done_buf == DONE_TRUNCATED
        return self.obs_buf, self.rew_buf, terminated, truncated, {}

    def save_state(self):
        return {
            "cars": self._cars_buf.clone(),
            "cars_int": wp.to_torch(self.cars_int).clone(),
            "obs": self.obs_buf.clone(),
            "rew": self.rew_buf.clone(),
            "done": self.done_buf.clone(),
        }

    def restore_state(self, s):
        self._cars_buf.copy_(s["cars"])
        wp.to_torch(self.cars_int).copy_(s["cars_int"])
        self.obs_buf.copy_(s["obs"])
        self.rew_buf.copy_(s["rew"])
        self.done_buf.copy_(s["done"])


def layer_init(layer: nn.Linear, std: float = np.sqrt(2.0)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.zeros_(layer.bias)
    return layer


class Agent(nn.Module):
    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        act_dim: int = ACT_DIM,
        obs_low: np.ndarray = OBS_LOW,
        obs_high: np.ndarray = OBS_HIGH,
    ):
        super().__init__()
        mid = (obs_low + obs_high) * 0.5
        half_range = (obs_high - obs_low) * 0.5
        self.register_buffer("obs_mid", torch.from_numpy(mid))
        self.register_buffer("obs_inv_scale", torch.from_numpy(1.0 / half_range))

        self.backbone = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, 64)),
            nn.ELU(),
        )
        self.actor_mean = layer_init(nn.Linear(64, act_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        self.critic = layer_init(nn.Linear(64, 1), std=1.0)

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.backbone((obs - self.obs_mid) * self.obs_inv_scale)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.features(obs)).squeeze(-1)

    def _dist(self, h: torch.Tensor) -> Normal:
        mean = torch.tanh(self.actor_mean(h))
        std = self.actor_logstd.exp().expand_as(mean)
        return Normal(mean, std)

    def act_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.features(obs)
        dist = self._dist(h)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(h).squeeze(-1)
        return action, log_prob, entropy, value

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor_mean(self.features(obs)))


class KLAdaptiveLR:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        target_kl: float = 0.01,
        factor: float = 1.5,
        min_lr: float = 1e-6,
        max_lr: float = 1e-2,
    ):
        self.optimizer = optimizer
        self.target = target_kl
        self.factor = factor
        self.min_lr = min_lr
        self.max_lr = max_lr

    def step(self, kl: float) -> None:
        for pg in self.optimizer.param_groups:
            lr = pg["lr"]
            if kl > 2.0 * self.target:
                pg["lr"] = max(self.min_lr, lr / self.factor)
            elif kl < 0.5 * self.target:
                pg["lr"] = min(self.max_lr, lr * self.factor)

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


def record_rollout(
    env: RacingEnv,
    agent: Agent,
    num_steps: int,
    output_path: Path,
    deterministic: bool = True,
) -> None:
    saved = env.save_state()
    was_training = agent.training
    agent.eval()
    try:
        m = env.map
        corners = np.array(
            [
                [-LENGTH / 2, -WIDTH / 2],
                [LENGTH / 2, -WIDTH / 2],
                [LENGTH / 2, WIDTH / 2],
                [-LENGTH / 2, WIDTH / 2],
            ]
        )

        def w2p(x, y):
            return int((x - m.ox) / m.res), int(m.h - 1 - (y - m.oy) / m.res)

        trail: deque = deque(maxlen=300)
        obs, _ = env.reset()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(
            str(output_path), fps=int(1 / DT), macro_block_size=1
        ) as writer:
            for _ in range(num_steps):
                with torch.no_grad():
                    if deterministic:
                        action = agent.deterministic(obs)
                    else:
                        action, *_ = agent.act_value(obs)
                obs, _, term, trunc, _ = env.step(action)

                x, y, psi = (float(v) for v in env._cars_buf[0, [0, 1, 4]])
                if bool(term[0].item()) or bool(trunc[0].item()):
                    trail.clear()
                trail.append((x, y))

                frame = cvtColor(m.raw, COLOR_GRAY2RGB)
                if len(trail) > 1:
                    polylines(
                        frame,
                        [np.array([w2p(*p) for p in trail], dtype=np.int32)],
                        False,
                        (0, 200, 0),
                        2,
                    )
                R = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
                world = corners @ R.T + (x, y)
                fillPoly(
                    frame,
                    [np.array([w2p(*p) for p in world], dtype=np.int32)],
                    (255, 50, 50),
                )
                writer.append_data(frame)
    finally:
        env.restore_state(saved)
        agent.train(was_training)


def _wandb_log(metrics: dict, step: int, video_path: Path | None = None) -> None:
    try:
        import wandb

        if wandb.run is None:
            return
        if video_path is not None:
            metrics = {
                **metrics,
                "rollout": wandb.Video(str(video_path), format="mp4"),
            }
        wandb.log(metrics, step=step)
    except Exception as exc:
        print(f"[warporacer] wandb log failed: {exc}")


def train(
    env: RacingEnv,
    agent: Agent,
    iterations: int = 2000,
    rollouts: int = 24,
    learning_epochs: int = 5,
    mini_batches: int = 4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    vf_clip_coef: float = 0.2,
    vf_coef: float = 2.0,
    ent_coef: float = 0.0,
    max_grad_norm: float = 0.5,
    learning_rate: float = 3e-4,
    target_kl: float = 0.01,
    log_dir: Path = Path("./logs"),
    record_every: int = 100,
    record_steps: int = 1800,
) -> float:
    device = next(agent.parameters()).device
    num_envs = env.num_envs
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    scheduler = KLAdaptiveLR(optimizer, target_kl=target_kl)

    obs_buf = torch.zeros((rollouts, num_envs, OBS_DIM), device=device)
    act_buf = torch.zeros((rollouts, num_envs, ACT_DIM), device=device)
    logp_buf = torch.zeros((rollouts, num_envs), device=device)
    rew_buf = torch.zeros((rollouts, num_envs), device=device)
    done_buf = torch.zeros((rollouts, num_envs), device=device)
    val_buf = torch.zeros((rollouts, num_envs), device=device)
    ep_ret_buf = torch.zeros((rollouts, num_envs), device=device)
    ep_len_buf = torch.zeros((rollouts, num_envs), device=device)

    obs, _ = env.reset()
    ep_returns = torch.zeros(num_envs, device=device)
    ep_steps = torch.zeros(num_envs, device=device)
    completed_returns: deque = deque(maxlen=100)
    completed_lengths: deque = deque(maxlen=100)

    global_step = 0
    t0 = time.time()
    last_t = t0

    for it in range(iterations):
        agent.eval()
        for t in range(rollouts):
            obs_buf[t] = obs
            with torch.no_grad():
                action, log_prob, _, value = agent.act_value(obs)
            act_buf[t] = action
            logp_buf[t] = log_prob
            val_buf[t] = value

            obs, reward, term, trunc, _ = env.step(action)
            done = (term | trunc).float()
            rew_buf[t] = reward
            done_buf[t] = done
            ep_returns += reward
            ep_steps += 1
            ep_ret_buf[t] = ep_returns
            ep_len_buf[t] = ep_steps
            nonterm = 1.0 - done
            ep_returns = ep_returns * nonterm
            ep_steps = ep_steps * nonterm

        global_step += rollouts * num_envs

        with torch.no_grad():
            next_value = agent.value(obs)
            adv_buf = torch.zeros_like(rew_buf)
            last_gae = torch.zeros_like(next_value)
            for t in reversed(range(rollouts)):
                next_v = val_buf[t + 1] if t < rollouts - 1 else next_value
                nonterm = 1.0 - done_buf[t]
                delta = rew_buf[t] + gamma * next_v * nonterm - val_buf[t]
                last_gae = delta + gamma * gae_lambda * nonterm * last_gae
                adv_buf[t] = last_gae
            ret_buf = adv_buf + val_buf

        finished = done_buf.bool()
        if finished.any():
            completed_returns.extend(ep_ret_buf[finished].cpu().tolist())
            completed_lengths.extend(ep_len_buf[finished].cpu().tolist())

        b_obs = obs_buf.reshape(-1, OBS_DIM)
        b_act = act_buf.reshape(-1, ACT_DIM)
        b_logp = logp_buf.reshape(-1)
        b_adv = adv_buf.reshape(-1)
        b_ret = ret_buf.reshape(-1)
        b_val = val_buf.reshape(-1)
        batch_size = b_obs.shape[0]
        mb_size = batch_size // mini_batches

        agent.train()
        kl_acc = clipfrac_acc = pg_acc = v_acc = ent_acc = 0.0
        n_updates = 0

        for epoch in range(learning_epochs):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mb_size):
                mb = perm[start : start + mb_size]
                _, new_logp, entropy, new_val = agent.act_value(b_obs[mb], b_act[mb])

                logratio = new_logp - b_logp[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()
                kl_acc += approx_kl.item()
                clipfrac_acc += clipfrac.item()

                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef),
                ).mean()

                if vf_clip_coef > 0:
                    v_clipped = b_val[mb] + (new_val - b_val[mb]).clamp(
                        -vf_clip_coef, vf_clip_coef
                    )
                    v_loss = (
                        0.5
                        * torch.max(
                            (new_val - b_ret[mb]).pow(2),
                            (v_clipped - b_ret[mb]).pow(2),
                        ).mean()
                    )
                else:
                    v_loss = 0.5 * (new_val - b_ret[mb]).pow(2).mean()

                ent = entropy.mean()
                loss = pg_loss + vf_coef * v_loss - ent_coef * ent

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

                pg_acc += pg_loss.item()
                v_acc += v_loss.item()
                ent_acc += ent.item()
                n_updates += 1

        mean_kl = kl_acc / n_updates
        scheduler.step(mean_kl)

        now = time.time()
        sps = int(rollouts * num_envs / max(now - last_t, 1e-9))
        last_t = now

        metrics = {
            "losses/policy_loss": pg_acc / n_updates,
            "losses/value_loss": v_acc / n_updates,
            "losses/entropy": ent_acc / n_updates,
            "losses/approx_kl": mean_kl,
            "losses/clipfrac": clipfrac_acc / n_updates,
            "policy/std": agent.actor_logstd.exp().mean().item(),
            "charts/learning_rate": scheduler.lr,
            "charts/sps": sps,
            "charts/iteration": it,
        }
        if completed_returns:
            metrics["charts/episodic_return"] = float(np.mean(completed_returns))
            metrics["charts/episodic_length"] = float(np.mean(completed_lengths))

        _wandb_log(metrics, global_step)
        if it % 10 == 0:
            er = metrics.get("charts/episodic_return", float("nan"))
            print(
                f"[iter {it:5d}] step={global_step:>10d} sps={sps:>6d} "
                f"ret={er:8.3f} kl={mean_kl:.4f} lr={scheduler.lr:.2e}"
            )

        if record_every > 0 and record_steps > 0 and (it + 1) % record_every == 0:
            out = log_dir / f"rollout_iter{it + 1:06d}.mp4"
            try:
                record_rollout(env, agent, record_steps, out)
                _wandb_log({}, global_step, video_path=out)
            except Exception as exc:
                print(f"[warporacer] rollout at iter {it + 1} failed: {exc}")

    return time.time() - t0


def main(
    map_yaml: Path,
    num_envs: int = 4096,
    iterations: int = 2000,
    seed: int = 0,
    log_dir: Path = Path("./logs"),
    device: str = "",
    record: int = 1800,
    record_every: int = 100,
    record_steps: int = 1800,
    use_wandb: bool = True,
):
    log_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    env = RacingEnv(map_yaml, num_envs=num_envs, seed=seed, device=device or None)
    agent = Agent().to(env.device)

    if use_wandb:
        try:
            import wandb

            wandb.init(
                project="warporacer",
                name=f"seed{seed}_n{num_envs}",
                config={
                    "num_envs": num_envs,
                    "iterations": iterations,
                    "seed": seed,
                    "map": str(map_yaml),
                    "obs_dim": OBS_DIM,
                    "act_dim": ACT_DIM,
                },
            )
        except Exception as exc:
            print(f"[warporacer] wandb init failed: {exc}")

    elapsed = train(
        env,
        agent,
        iterations=iterations,
        log_dir=log_dir,
        record_every=record_every,
        record_steps=record_steps,
    )
    print(
        f"[warporacer] training finished in {elapsed:.1f}s "
        f"({env.nan_events} sanitized NaN events)"
    )

    torch.save(agent.state_dict(), log_dir / "agent_final.pt")

    if record > 0:
        out = log_dir / "rollout.mp4"
        record_rollout(env, agent, record, out)
        _wandb_log({}, iterations * 24 * num_envs, video_path=out)


if __name__ == "__main__":
    run(main)
