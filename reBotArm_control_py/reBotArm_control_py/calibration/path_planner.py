"""标定路径规划：种子位姿 → 无碰撞 waypoint 序列。"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .io import PathPlan, PoseReport, SeedConfig, Waypoint
from .pose_validator import PoseValidator


def _subdivide_segment(
    q_a: np.ndarray,
    q_b: np.ndarray,
    max_dq: float,
) -> List[np.ndarray]:
    dq = np.abs(q_b - q_a)
    n = int(np.ceil(np.max(dq) / max(max_dq, 1e-6)))
    n = max(n, 1)
    points = []
    for k in range(1, n):
        alpha = k / n
        points.append((1.0 - alpha) * q_a + alpha * q_b)
    return points


def _append_segment(
    validator: PoseValidator,
    waypoints: List[Waypoint],
    q_from: np.ndarray,
    q_to: np.ndarray,
    pose_name: str,
    kind: str,
    segment_samples: int,
    max_segment_dq: float,
    min_clearance: float,
    direction: Optional[str] = None,
) -> Tuple[bool, str, float]:
    """从 q_from 安全连接到 q_to，必要时插入 transit 点。"""
    ok, alpha, reason, md = validator.check_segment(
        q_from, q_to, n_samples=segment_samples
    )
    min_clearance = min(min_clearance, md)
    if ok:
        suffix = f"_{direction}" if direction else ""
        waypoints.append(
            Waypoint(
                name=f"{pose_name}_{kind}{suffix}",
                kind=kind,
                pose_name=pose_name,
                q=q_to.copy(),
                direction=direction,
            )
        )
        return True, "", min_clearance

    # 尝试细分
    mids = _subdivide_segment(q_from, q_to, max_segment_dq)
    q_curr = q_from.copy()
    for i, q_mid in enumerate(mids):
        ok, alpha, reason, md = validator.check_segment(
            q_curr, q_mid, n_samples=segment_samples
        )
        min_clearance = min(min_clearance, md)
        if not ok:
            return (
                False,
                f"段内碰撞/越界 @ 中转 {i}: {reason} (alpha={alpha:.2f})",
                min_clearance,
            )
        waypoints.append(
            Waypoint(
                name=f"{pose_name}_transit_{i}",
                kind="transit",
                pose_name=pose_name,
                q=q_mid.copy(),
            )
        )
        q_curr = q_mid

    ok, alpha, reason, md = validator.check_segment(
        q_curr, q_to, n_samples=segment_samples
    )
    min_clearance = min(min_clearance, md)
    if not ok:
        return False, f"末段失败: {reason} (alpha={alpha:.2f})", min_clearance

    suffix = f"_{direction}" if direction else ""
    wp = Waypoint(
        name=f"{pose_name}_{kind}{suffix}",
        kind=kind,
        pose_name=pose_name,
        q=q_to.copy(),
        direction=direction,
    )
    waypoints.append(wp)
    return True, "", min_clearance


def plan_calibration_path(
    config: SeedConfig,
    validator: PoseValidator,
    urdf_path: str,
    urdf_sha256: str,
) -> PathPlan:
    home_q = config.home_q.copy()
    reports: List[PoseReport] = []
    waypoints: List[Waypoint] = []
    min_clearance = float("inf")
    total_dq = 0.0
    all_accepted = True

    ok, reason, md = validator.check_static(home_q)
    min_clearance = min(min_clearance, md)
    if not ok:
        raise RuntimeError(f"home 位姿非法: {reason}")

    waypoints.append(
        Waypoint(name="home", kind="home", pose_name="home", q=home_q.copy())
    )
    q_prev = home_q.copy()

    for pose in config.poses:
        q_t = pose.q.copy()
        delta = pose.approach_delta.copy()
        q_minus = q_t - delta
        q_plus = q_t + delta

        pose_ok = True
        fail_reason = ""
        for label, q_chk in (
            ("target", q_t),
            ("approach_minus", q_minus),
            ("approach_plus", q_plus),
        ):
            ok, reason, md = validator.check_static(q_chk)
            min_clearance = min(min_clearance, md)
            if not ok:
                pose_ok = False
                fail_reason = f"{label}: {reason}"
                break

        if not pose_ok:
            reports.append(PoseReport(name=pose.name, accepted=False, reason=fail_reason))
            all_accepted = False
            continue

        reports.append(PoseReport(name=pose.name, accepted=True, reason=""))

        # 路径: ... -> approach_minus -> target (minus) -> approach_plus -> target (plus)
        sequence = [
            ("approach_minus", q_minus, "minus"),
            ("target", q_t, "minus"),
            ("approach_plus", q_plus, "plus"),
            ("target", q_t, "plus"),
        ]

        for kind, q_goal, direction in sequence:
            ok, reason, min_clearance = _append_segment(
                validator,
                waypoints,
                q_prev,
                q_goal,
                pose.name,
                kind,
                config.segment_samples,
                config.max_segment_dq,
                min_clearance,
                direction=direction if kind == "target" else None,
            )
            if not ok:
                reports[-1] = PoseReport(
                    name=pose.name,
                    accepted=False,
                    reason=f"路径段失败: {reason}",
                )
                all_accepted = False
                break
            total_dq += float(np.sum(np.abs(q_goal - q_prev)))
            q_prev = q_goal.copy()

    # 最后回到 home
    if all_accepted:
        ok, reason, min_clearance = _append_segment(
            validator,
            waypoints,
            q_prev,
            home_q,
            "home",
            "home",
            config.segment_samples,
            config.max_segment_dq,
            min_clearance,
        )
        if not ok:
            all_accepted = False
            reports.append(
                PoseReport(name="return_home", accepted=False, reason=reason)
            )
        else:
            total_dq += float(np.sum(np.abs(home_q - q_prev)))

    return PathPlan(
        urdf_path=urdf_path,
        urdf_sha256=urdf_sha256,
        home_q=home_q,
        waypoints=waypoints,
        pose_reports=reports,
        segment_samples=config.segment_samples,
        max_segment_dq=config.max_segment_dq,
        limit_margin=config.limit_margin,
        settle_time_s=config.settle_time_s,
        sample_time_s=config.sample_time_s,
        move_duration_s=config.move_duration_s,
        max_velocity_rad_s=config.max_velocity_rad_s,
        max_acceleration_rad_s2=config.max_acceleration_rad_s2,
        mit_kp=config.mit_kp,
        mit_kd=config.mit_kd,
        min_clearance_m=min_clearance,
        total_path_dq=total_dq,
        collision_check_enabled=validator.collision_enabled,
    )
