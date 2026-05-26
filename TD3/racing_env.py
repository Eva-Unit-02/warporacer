from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import warp as wp

try:
    from .config import (
        ACCEL_MAX,
        ACT_DIM,
        CAR_HALF_DIAG,
        CAR_LENGTH,
        CAR_WIDTH,
        DONE_TERMINATED,
        DONE_TRUNCATED,
        DRIFT_FRACTION,
        DT_HALF_SUBSTEP,
        DT_SIXTH_SUBSTEP,
        DT_SUBSTEP,
        GRAVITY,
        LF,
        LIDAR_FOV,
        LIDAR_RANGE,
        LR,
        MAX_EPISODE_STEPS,
        MU,
        NUM_LIDAR,
        NUM_LOOKAHEAD,
        OBS_DIM,
        OBS_FRENET_OFFSET,
        OBS_LOOKAHEAD_OFFSET,
        PROGRESS_BACK_WINDOW,
        PROGRESS_FORWARD_WINDOW,
        PROGRESS_REWARD_SCALE,
        PROGRESS_SPEED_SCALE,
        SLIP_ANGLE_MAX,
        STALL_PENALTY,
        STEER_MAX,
        STEER_MIN,
        STEER_RATE_MAX,
        SUBSTEPS,
        TERMINATION_PENALTY,
        VEL_MAX,
        VEL_MIN,
        YAW_RATE_MAX,
    )
    from .map_processing import Map
except ImportError:
    from config import (
        ACCEL_MAX,
        ACT_DIM,
        CAR_HALF_DIAG,
        CAR_LENGTH,
        CAR_WIDTH,
        DONE_TERMINATED,
        DONE_TRUNCATED,
        DRIFT_FRACTION,
        DT_HALF_SUBSTEP,
        DT_SIXTH_SUBSTEP,
        DT_SUBSTEP,
        GRAVITY,
        LF,
        LIDAR_FOV,
        LIDAR_RANGE,
        LR,
        MAX_EPISODE_STEPS,
        MU,
        NUM_LIDAR,
        NUM_LOOKAHEAD,
        OBS_DIM,
        OBS_FRENET_OFFSET,
        OBS_LOOKAHEAD_OFFSET,
        PROGRESS_BACK_WINDOW,
        PROGRESS_FORWARD_WINDOW,
        PROGRESS_REWARD_SCALE,
        PROGRESS_SPEED_SCALE,
        SLIP_ANGLE_MAX,
        STALL_PENALTY,
        STEER_MAX,
        STEER_MIN,
        STEER_RATE_MAX,
        SUBSTEPS,
        TERMINATION_PENALTY,
        VEL_MAX,
        VEL_MIN,
        YAW_RATE_MAX,
    )
    from map_processing import Map


@wp.struct
class VehicleDerivative:
    dx: float
    dy: float
    dpsi: float
    dpsi_rate: float
    dbeta: float
    dvel: float


@wp.func
def bicycle_derivative(
    steer: float,
    velocity: float,
    heading: float,
    yaw_rate: float,
    slip_angle: float,
    steer_rate: float,
    acceleration: float,
    mu_scale: float,
    mass_scale: float,
    lf_scale: float,
    lr_scale: float,
) -> VehicleDerivative:
    front = LF * lf_scale
    rear = LR * lr_scale
    wheelbase = front + rear
    grip = MU * mu_scale
    max_accel = grip * GRAVITY

    curvature_yaw = velocity * wp.tan(steer) / wheelbase
    capped_yaw = max_accel / wp.max(wp.abs(velocity), 0.5)
    yaw = wp.clamp(curvature_yaw, -capped_yaw, capped_yaw)

    lateral_accel = velocity * yaw
    longitudinal_cap = wp.sqrt(wp.max(max_accel * max_accel - lateral_accel * lateral_accel, 0.0))

    cos_h = wp.cos(heading)
    sin_h = wp.sin(heading)
    out = VehicleDerivative()
    out.dx = velocity * cos_h
    out.dy = velocity * sin_h
    out.dpsi = yaw
    out.dvel = wp.clamp(acceleration, -longitudinal_cap, longitudinal_cap)
    out.dpsi_rate = 0.0
    out.dbeta = 0.0
    return out


@wp.func
def integrate_vehicle(
    steer: float,
    velocity: float,
    heading: float,
    yaw_rate: float,
    slip_angle: float,
    steer_rate: float,
    acceleration: float,
    mu_scale: float,
    mass_scale: float,
    lf_scale: float,
    lr_scale: float,
) -> VehicleDerivative:
    half_delta = steer_rate * DT_HALF_SUBSTEP
    full_delta = steer_rate * DT_SUBSTEP

    k1 = bicycle_derivative(
        steer,
        velocity,
        heading,
        yaw_rate,
        slip_angle,
        steer_rate,
        acceleration,
        mu_scale,
        mass_scale,
        lf_scale,
        lr_scale,
    )
    k2 = bicycle_derivative(
        steer + half_delta,
        velocity + k1.dvel * DT_HALF_SUBSTEP,
        heading + k1.dpsi * DT_HALF_SUBSTEP,
        yaw_rate + k1.dpsi_rate * DT_HALF_SUBSTEP,
        slip_angle + k1.dbeta * DT_HALF_SUBSTEP,
        steer_rate,
        acceleration,
        mu_scale,
        mass_scale,
        lf_scale,
        lr_scale,
    )
    k3 = bicycle_derivative(
        steer + half_delta,
        velocity + k2.dvel * DT_HALF_SUBSTEP,
        heading + k2.dpsi * DT_HALF_SUBSTEP,
        yaw_rate + k2.dpsi_rate * DT_HALF_SUBSTEP,
        slip_angle + k2.dbeta * DT_HALF_SUBSTEP,
        steer_rate,
        acceleration,
        mu_scale,
        mass_scale,
        lf_scale,
        lr_scale,
    )
    k4 = bicycle_derivative(
        steer + full_delta,
        velocity + k3.dvel * DT_SUBSTEP,
        heading + k3.dpsi * DT_SUBSTEP,
        yaw_rate + k3.dpsi_rate * DT_SUBSTEP,
        slip_angle + k3.dbeta * DT_SUBSTEP,
        steer_rate,
        acceleration,
        mu_scale,
        mass_scale,
        lf_scale,
        lr_scale,
    )

    result = VehicleDerivative()
    result.dx = (k1.dx + 2.0 * (k2.dx + k3.dx) + k4.dx) * DT_SIXTH_SUBSTEP
    result.dy = (k1.dy + 2.0 * (k2.dy + k3.dy) + k4.dy) * DT_SIXTH_SUBSTEP
    result.dpsi = (k1.dpsi + 2.0 * (k2.dpsi + k3.dpsi) + k4.dpsi) * DT_SIXTH_SUBSTEP
    result.dvel = (k1.dvel + 2.0 * (k2.dvel + k3.dvel) + k4.dvel) * DT_SIXTH_SUBSTEP
    result.dpsi_rate = (
        k1.dpsi_rate + 2.0 * (k2.dpsi_rate + k3.dpsi_rate) + k4.dpsi_rate
    ) * DT_SIXTH_SUBSTEP
    result.dbeta = (k1.dbeta + 2.0 * (k2.dbeta + k3.dbeta) + k4.dbeta) * DT_SIXTH_SUBSTEP
    return result


@wp.kernel
def advance_kernel(
    actions: wp.array(dtype=wp.vec2),
    observations: wp.array2d(dtype=wp.float32),
    rewards: wp.array(dtype=wp.float32),
    done_codes: wp.array(dtype=wp.int32),
    cars: wp.array2d(dtype=wp.float32),
    car_meta: wp.array2d(dtype=wp.int32),
    domain_randomization: wp.array2d(dtype=wp.float32),
    map_origin: wp.vec2,
    map_resolution: float,
    distance_map: wp.array2d(dtype=wp.float32),
    nearest_waypoint: wp.array2d(dtype=wp.int32),
    centerline: wp.array(dtype=wp.vec3),
    num_waypoints: int,
    track_length: float,
    lookahead_stride: int,
    lidar_basis: wp.array(dtype=wp.vec2),
    seed_base: int,
):
    env_id = wp.tid()
    x = cars[env_id, 0]
    y = cars[env_id, 1]
    prev_x = x
    prev_y = y
    steer = cars[env_id, 2]
    velocity = cars[env_id, 3]
    heading = cars[env_id, 4]
    yaw_rate = cars[env_id, 5]
    slip_angle = cars[env_id, 6]
    step_count = car_meta[env_id, 0]
    waypoint_index = car_meta[env_id, 1]

    mu_scale = domain_randomization[env_id, 0]
    mass_scale = domain_randomization[env_id, 1]
    lf_scale = domain_randomization[env_id, 2]
    lr_scale = domain_randomization[env_id, 3]

    map_width = distance_map.shape[0]
    map_height = distance_map.shape[1]
    map_height_f = wp.float32(map_height) - 1.0

    steer_rate = wp.clamp(actions[env_id][0], -1.0, 1.0) * STEER_RATE_MAX
    if (steer_rate < 0.0 and steer <= STEER_MIN) or (steer_rate > 0.0 and steer >= STEER_MAX):
        steer_rate = 0.0

    accel = wp.clamp(actions[env_id][1], -1.0, 1.0) * ACCEL_MAX
    if (accel < 0.0 and velocity <= VEL_MIN) or (accel > 0.0 and velocity >= VEL_MAX):
        accel = 0.0

    steer_delta = steer_rate * DT_SUBSTEP
    for _ in range(SUBSTEPS):
        delta = integrate_vehicle(
            steer,
            velocity,
            heading,
            yaw_rate,
            slip_angle,
            steer_rate,
            accel,
            mu_scale,
            mass_scale,
            lf_scale,
            lr_scale,
        )
        x += delta.dx
        y += delta.dy
        steer += steer_delta
        velocity += delta.dvel
        heading += delta.dpsi
        yaw_rate += delta.dpsi_rate
        slip_angle += delta.dbeta

    steer = wp.clamp(steer, STEER_MIN, STEER_MAX)
    velocity = wp.clamp(velocity, VEL_MIN, VEL_MAX)
    yaw_rate = wp.clamp(yaw_rate, -YAW_RATE_MAX, YAW_RATE_MAX)
    slip_angle = wp.clamp(slip_angle, -SLIP_ANGLE_MAX, SLIP_ANGLE_MAX)

    cell_x = wp.clamp(wp.int32((x - map_origin[0]) / map_resolution), 0, map_width - 1)
    cell_y = wp.clamp(wp.int32(map_height_f - (y - map_origin[1]) / map_resolution), 0, map_height - 1)
    wall_distance = distance_map[cell_x, cell_y] * map_resolution
    terminated = wall_distance < CAR_HALF_DIAG
    truncated = step_count >= MAX_EPISODE_STEPS
    step_count += 1

    current_waypoint = waypoint_index
    best_dist_sq = float(3.4028234663852886e38)
    for offset in range(-PROGRESS_BACK_WINDOW, PROGRESS_FORWARD_WINDOW + 1):
        candidate = waypoint_index + offset
        if candidate < 0:
            candidate += num_waypoints
        elif candidate >= num_waypoints:
            candidate -= num_waypoints

        center = centerline[candidate]
        dx_center = x - center[0]
        dy_center = y - center[1]
        dist_sq = dx_center * dx_center + dy_center * dy_center
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            current_waypoint = candidate

    waypoint_delta = current_waypoint - waypoint_index
    if 2 * waypoint_delta > num_waypoints:
        waypoint_delta -= num_waypoints
    elif 2 * waypoint_delta < -num_waypoints:
        waypoint_delta += num_waypoints
    waypoint_delta = wp.clamp(waypoint_delta, -PROGRESS_BACK_WINDOW, PROGRESS_FORWARD_WINDOW)

    tangent = centerline[current_waypoint][2]
    forward_speed = velocity * wp.cos(slip_angle + heading - tangent)
    prev_tangent = centerline[waypoint_index][2]
    step_forward_distance = (x - prev_x) * wp.cos(prev_tangent) + (y - prev_y) * wp.sin(prev_tangent)
    progress_reward = (
        step_forward_distance
        / track_length
        * PROGRESS_REWARD_SCALE
        * (1.0 + wp.max(forward_speed, 0.0) / PROGRESS_SPEED_SCALE)
    )
    stall_cost = wp.where(step_forward_distance <= 0.0, -STALL_PENALTY, 0.0)
    rewards[env_id] = progress_reward + stall_cost + wp.where(terminated, -TERMINATION_PENALTY, 0.0)

    if terminated:
        done_codes[env_id] = DONE_TERMINATED
    elif truncated:
        done_codes[env_id] = DONE_TRUNCATED
    else:
        done_codes[env_id] = 0

    if terminated or truncated:
        rng = wp.rand_init(seed_base + env_id * 97 + step_count * 29 + current_waypoint * 13)
        reset_index = wp.int32(wp.randf(rng) * wp.float32(num_waypoints)) % num_waypoints
        pose = centerline[reset_index]
        x = pose[0]
        y = pose[1]
        heading = pose[2]
        steer = 0.0
        velocity = 0.0
        yaw_rate = 0.0
        slip_angle = 0.0
        step_count = 0
        current_waypoint = reset_index
        domain_randomization[env_id, 0] = 1.0 - DRIFT_FRACTION + 2.0 * DRIFT_FRACTION * wp.randf(rng)
        domain_randomization[env_id, 1] = 1.0 - DRIFT_FRACTION + 2.0 * DRIFT_FRACTION * wp.randf(rng)
        domain_randomization[env_id, 2] = 1.0 - DRIFT_FRACTION + 2.0 * DRIFT_FRACTION * wp.randf(rng)
        domain_randomization[env_id, 3] = 1.0 - DRIFT_FRACTION + 2.0 * DRIFT_FRACTION * wp.randf(rng)

    sin_heading = wp.sin(heading)
    cos_heading = wp.cos(heading)

    lidar_x = x + LF * cos_heading
    lidar_y = y + LF * sin_heading
    lidar_px = wp.clamp(wp.int32((lidar_x - map_origin[0]) / map_resolution), 0, map_width - 1)
    lidar_py = wp.clamp(wp.int32(map_height_f - (lidar_y - map_origin[1]) / map_resolution), 0, map_height - 1)
    lidar_origin_px = wp.vec2(wp.float32(lidar_px), wp.float32(lidar_py))
    lidar_range_px = LIDAR_RANGE / map_resolution

    for ray_idx in range(lidar_basis.shape[0]):
        basis_cos = lidar_basis[ray_idx][0]
        basis_sin = lidar_basis[ray_idx][1]
        direction = wp.vec2(
            cos_heading * basis_cos - sin_heading * basis_sin,
            -(sin_heading * basis_cos + cos_heading * basis_sin),
        )
        ray = lidar_origin_px
        distance_px = float(0.0)
        while distance_px < lidar_range_px:
            sample_x = wp.int32(ray[0])
            sample_y = wp.int32(ray[1])
            if sample_x < 0 or sample_x >= map_width or sample_y < 0 or sample_y >= map_height:
                break
            jump = distance_map[sample_x, sample_y]
            ray = ray + direction * jump
            distance_px += jump
            if jump == 0.0:
                break
        observations[env_id, 3 + ray_idx] = wp.min(distance_px, lidar_range_px) * map_resolution

    waypoint_pose = centerline[current_waypoint]
    waypoint_heading = waypoint_pose[2]
    sin_wp = wp.sin(waypoint_heading)
    cos_wp = wp.cos(waypoint_heading)
    heading_error = wp.atan2(
        sin_wp * cos_heading - cos_wp * sin_heading,
        cos_wp * cos_heading + sin_wp * sin_heading,
    )
    lateral_error = -(x - waypoint_pose[0]) * sin_wp + (y - waypoint_pose[1]) * cos_wp
    observations[env_id, OBS_FRENET_OFFSET] = heading_error
    observations[env_id, OBS_FRENET_OFFSET + 1] = lateral_error

    look_idx = current_waypoint
    for offset in range(NUM_LOOKAHEAD):
        look_idx += lookahead_stride
        if look_idx >= num_waypoints:
            look_idx -= num_waypoints
        waypoint = centerline[look_idx]
        delta_x = waypoint[0] - x
        delta_y = waypoint[1] - y
        observations[env_id, OBS_LOOKAHEAD_OFFSET + offset * 2] = delta_x * cos_heading + delta_y * sin_heading
        observations[env_id, OBS_LOOKAHEAD_OFFSET + offset * 2 + 1] = (
            -delta_x * sin_heading + delta_y * cos_heading
        )

    observations[env_id, 0] = steer
    observations[env_id, 1] = velocity
    observations[env_id, 2] = yaw_rate

    cars[env_id, 0] = x
    cars[env_id, 1] = y
    cars[env_id, 2] = steer
    cars[env_id, 3] = velocity
    cars[env_id, 4] = heading
    cars[env_id, 5] = yaw_rate
    cars[env_id, 6] = slip_angle
    car_meta[env_id, 0] = step_count
    car_meta[env_id, 1] = current_waypoint


class RacingEnv:
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)

    def __init__(self, map_yaml: Path, num_envs: int, seed: int = 0, device: str | None = None):
        wp.init()
        self.num_envs = num_envs
        self.seed = int(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.map = Map(map_yaml)
        self.lookahead_stride = self.map.lookahead_stride

        wp_device = self.device
        distance_map = self.map.distance_field.T.astype(np.float32)
        waypoint_lut = self.map.closest_waypoint.T.astype(np.int32)
        centerline = np.column_stack((self.map.centerline, self.map.heading)).astype(np.float32)

        self.distance_map = wp.array(distance_map, dtype=float, device=wp_device)
        self.waypoint_lut = wp.array(waypoint_lut, dtype=int, device=wp_device)
        self.centerline = wp.array(centerline, dtype=wp.vec3, device=wp_device)
        self.num_waypoints = len(self.map.centerline)

        rng = np.random.default_rng(seed)
        start_idx = rng.integers(0, self.num_waypoints, size=num_envs)
        car_state = np.zeros((num_envs, 7), dtype=np.float32)
        car_state[:, 0] = self.map.centerline[start_idx, 0]
        car_state[:, 1] = self.map.centerline[start_idx, 1]
        car_state[:, 4] = self.map.heading[start_idx]

        car_meta = np.zeros((num_envs, 2), dtype=np.int32)
        car_meta[:, 1] = start_idx
        randomized = 1.0 - DRIFT_FRACTION + 2.0 * DRIFT_FRACTION * rng.random((num_envs, 4), dtype=np.float32)

        self.car_state = wp.array(car_state, dtype=float, device=wp_device)
        self.car_meta = wp.array(car_meta, dtype=int, device=wp_device)
        self.randomized_params = wp.array(randomized, dtype=float, device=wp_device)
        self.observations = wp.zeros((num_envs, OBS_DIM), dtype=float, device=wp_device)
        self.rewards = wp.zeros(num_envs, dtype=float, device=wp_device)
        self.done_codes = wp.zeros(num_envs, dtype=int, device=wp_device)

        self.obs_torch = wp.to_torch(self.observations)
        self.rew_torch = wp.to_torch(self.rewards)
        self.done_torch = wp.to_torch(self.done_codes)
        self.car_state_torch = wp.to_torch(self.car_state)
        self.car_meta_torch = wp.to_torch(self.car_meta)
        self._step_counts = self.car_meta_torch[:, 0]

        lidar_angles = np.linspace(-LIDAR_FOV * 0.5, LIDAR_FOV * 0.5, NUM_LIDAR, dtype=np.float32)
        lidar_dirs = np.column_stack((np.cos(lidar_angles), np.sin(lidar_angles))).astype(np.float32)
        self.lidar_dirs = wp.array(lidar_dirs, dtype=wp.vec2, device=wp_device)

        self._zero_actions = wp.zeros(num_envs, dtype=wp.vec2, device=wp_device)
        self._launch_count = 0
        self._launch(self._zero_actions)
        self._sanitize_buffers()
        self._step_counts.zero_()
        self.rew_torch.zero_()
        self.done_torch.zero_()

    def _launch(self, action_buffer):
        launch_seed = (self.seed * 2654435761 + self._launch_count * 40503) & 0x7FFFFFFF
        wp.launch(
            advance_kernel,
            dim=self.num_envs,
            inputs=[
                action_buffer,
                self.observations,
                self.rewards,
                self.done_codes,
                self.car_state,
                self.car_meta,
                self.randomized_params,
                wp.vec2(self.map.origin_x, self.map.origin_y),
                self.map.resolution,
                self.distance_map,
                self.waypoint_lut,
                self.centerline,
                self.num_waypoints,
                self.map.track_length,
                self.lookahead_stride,
                self.lidar_dirs,
                int(launch_seed),
            ],
        )
        wp.synchronize_device(self.car_state.device)
        self._launch_count += 1

    def _sanitize_buffers(self):
        invalid = ~(torch.isfinite(self.obs_torch).all(dim=1) & torch.isfinite(self.car_state_torch).all(dim=1))
        if not invalid.any():
            return
        torch.nan_to_num_(self.obs_torch, nan=0.0, posinf=LIDAR_RANGE, neginf=0.0)
        torch.nan_to_num_(self.car_state_torch, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nan_to_num_(self.rew_torch, nan=0.0, posinf=0.0, neginf=0.0)
        self._step_counts[invalid] = MAX_EPISODE_STEPS
        self.done_torch[invalid] = DONE_TRUNCATED

    def reset(self):
        self._step_counts.fill_(MAX_EPISODE_STEPS)
        self._launch(self._zero_actions)
        self._sanitize_buffers()
        self._step_counts.zero_()
        self.rew_torch.zero_()
        self.done_torch.zero_()
        return self.obs_torch, {}

    def step(self, action: torch.Tensor):
        action_buffer = wp.from_torch(action.detach().contiguous(), dtype=wp.vec2)
        self._launch(action_buffer)
        self._sanitize_buffers()
        return (
            self.obs_torch,
            self.rew_torch,
            self.done_torch == DONE_TERMINATED,
            self.done_torch == DONE_TRUNCATED,
            {},
        )

    def save_state(self):
        snapshot = {
            "car_state": self.car_state_torch.clone(),
            "car_meta": self.car_meta_torch.clone(),
            "observations": self.obs_torch.clone(),
            "rewards": self.rew_torch.clone(),
            "done_codes": self.done_torch.clone(),
            "randomized_params": wp.to_torch(self.randomized_params).clone(),
            "launch_count": self._launch_count,
        }
        return snapshot

    def restore_state(self, snapshot):
        self.car_state_torch.copy_(snapshot["car_state"])
        self.car_meta_torch.copy_(snapshot["car_meta"])
        self.obs_torch.copy_(snapshot["observations"])
        self.rew_torch.copy_(snapshot["rewards"])
        self.done_torch.copy_(snapshot["done_codes"])
        wp.to_torch(self.randomized_params).copy_(snapshot["randomized_params"])
        self._launch_count = snapshot["launch_count"]
