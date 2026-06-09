"""reBot-DevArm 轨迹采样模块。

提供 SE(3) 测地线插值、三种时间剖面的离散采样，输出笛卡尔轨迹点。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pinocchio as pin


class TrajProfile(enum.Enum):
    LINEAR = "linear"
    MIN_JERK = "min_jerk"
    TRAPEZOID = "trapezoid"
    DOUBLE_S = "double_s"   # jerk 限制双 S 速度曲线（7 段）


@dataclass
class TrajPlanParams:
    dt: float = 0.005   # 200 Hz 插值
    profile: TrajProfile = TrajProfile.DOUBLE_S
    accel_ratio: float = 0.25   # 加速段时长占比 Ta/T（每侧）
    jerk_ratio: float = 0.5     # 加加速段占加速段比例 Tj/Ta，∈(0, 0.5]


@dataclass
class CartesianPoint:
    time: float
    pose: pin.SE3


@dataclass
class CartesianTrajectory:
    points_: List[CartesianPoint] = field(default_factory=list)

    def add_point(self, t: float, pose: pin.SE3) -> None:
        self.points_.append(CartesianPoint(t, pose))

    def duration(self) -> float:
        return self.points_[-1].time if self.points_ else 0.0

    def points(self) -> List[CartesianPoint]:
        return self.points_


@dataclass
class CartesianTrajectoryResult:
    trajectory: CartesianTrajectory
    n_points: int


def _double_s_profile(t: float, accel_ratio: float, jerk_ratio: float) -> float:
    """归一化双 S（jerk 限制）位置剖面 s(t)，t∈[0,1]。

    7 段对称结构（加加速→匀加速→减加速→匀速→加减速→匀减速→减减速），
    加速度连续、jerk 分段恒定有界 —— 即标准 double-S / S-curve。
    归一化到总时长 1、总位移 1：

      - ``Ta``: 加速段时长（每侧），= accel_ratio
      - ``Tj``: 加加速段时长，= jerk_ratio * Ta
      - 峰值速度  v = 1 / (1 - Ta)
      - 峰值加速度 a = v / (Ta - Tj)
      - jerk      j = a / Tj
    """
    t = max(0.0, min(1.0, t))
    Ta = max(0.01, min(0.5, accel_ratio))
    Tj = max(1e-3, min(0.5, jerk_ratio)) * Ta
    Tc = Ta - 2.0 * Tj          # 匀加速段时长（jerk_ratio=0.5 时为 0）
    v = 1.0 / (1.0 - Ta)        # 峰值速度
    a = v / (Ta - Tj)           # 峰值加速度
    j = a / Tj                  # 加加速度

    def _half(tt: float) -> float:
        """前半程（含至中点匀速）位置，tt∈[0, 0.5]。"""
        if tt <= Tj:                                   # 段1：jerk = +j
            return j * tt ** 3 / 6.0
        v1 = 0.5 * j * Tj * Tj
        p1 = j * Tj ** 3 / 6.0
        if tt <= Ta - Tj:                              # 段2：匀加速 a
            dt = tt - Tj
            return p1 + v1 * dt + 0.5 * a * dt * dt
        v2 = v1 + a * Tc
        p2 = p1 + v1 * Tc + 0.5 * a * Tc * Tc
        if tt <= Ta:                                   # 段3：jerk = -j
            dt = tt - (Ta - Tj)
            return p2 + v2 * dt + 0.5 * a * dt * dt - j * dt ** 3 / 6.0
        p3 = p2 + v2 * Tj + 0.5 * a * Tj * Tj - j * Tj ** 3 / 6.0
        return p3 + v * (tt - Ta)                      # 段4：匀速

    if t <= 0.5:
        return _half(t)
    return 1.0 - _half(1.0 - t)                        # 后半程点对称


def _apply_profile(
    t: float, profile: TrajProfile, accel_ratio: float, jerk_ratio: float = 0.5,
) -> float:
    """归一化时间 t∈[0,1] 经时间剖面映射到 s∈[0,1]。"""
    t = max(0.0, min(1.0, t))
    if profile == TrajProfile.LINEAR:
        return t
    if profile == TrajProfile.MIN_JERK:
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        return 10.0 * t3 - 15.0 * t4 + 6.0 * t5
    if profile == TrajProfile.TRAPEZOID:
        ta = max(0.01, min(0.49, accel_ratio))
        vm = 1.0 / (1.0 - ta)   # 峰值速度（位移归一化，端点连续）
        if t <= ta:
            return 0.5 * vm / ta * t * t
        if t <= 1.0 - ta:
            return 0.5 * vm * ta + vm * (t - ta)
        dt = 1.0 - t
        return 1.0 - 0.5 * vm / ta * dt * dt
    if profile == TrajProfile.DOUBLE_S:
        return _double_s_profile(t, accel_ratio, jerk_ratio)
    return t


def _se3_interpolate(a, b, s) -> pin.SE3:
    """SE(3) 测地线插值：s∈[0,1] 从 a 到 b。接受 SE3 对象或 (4,4) ndarray。"""
    if isinstance(a, np.ndarray):
        a = pin.SE3(a)
    if isinstance(b, np.ndarray):
        b = pin.SE3(b)
    return a * pin.exp6(pin.log6(a.inverse() * b) * s)


def plan_cartesian_geodesic_trajectory(
    start_pose: pin.SE3,
    end_pose: pin.SE3,
    duration: float,
    params: TrajPlanParams | None = None,
) -> CartesianTrajectoryResult:
    """采样 SE(3) 测地线路径。

    参数:
        start_pose: 起始位姿。
        end_pose:   终止位姿。
        duration:   总时长（秒），必须 > 0。
        params:     采样参数（默认 :class:`TrajPlanParams`）。

    返回:
        :class:`CartesianTrajectoryResult`。
    """
    if duration <= 0.0:
        raise ValueError("duration 必须 > 0")
    if params is None:
        params = TrajPlanParams()

    traj = CartesianTrajectory()
    n = max(2, int(np.ceil(duration / params.dt)) + 1)
    dt = duration / (n - 1)

    for i in range(n):
        t = i * dt
        s = _apply_profile(
            t / duration, params.profile, params.accel_ratio, params.jerk_ratio,
        )
        traj.add_point(t, _se3_interpolate(start_pose, end_pose, s))

    return CartesianTrajectoryResult(trajectory=traj, n_points=n)
