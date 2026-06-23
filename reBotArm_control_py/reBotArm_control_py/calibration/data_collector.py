"""程序 2：沿标定路径 MoveJ + 静止采样。"""

from __future__ import annotations

import signal
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .io import PathPlan, TargetSample, Waypoint


def _quintic_alpha(s: float) -> float:
    """5 次多项式位置插值系数: 10s^3 - 15s^4 + 6s^5（端点速度/加速度为 0）。"""
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def _quintic_alpha_dot(s: float) -> float:
    """5 次多项式速度系数（对归一化时间 s 求导）: 30s^2 - 60s^3 + 30s^4。"""
    return 30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4


def move_j(
    arm,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    duration: float,
    kp: np.ndarray,
    kd: np.ndarray,
    max_vel: float,
    max_acc: float,
) -> np.ndarray:
    """关节空间 MoveJ：五次多项式位置 + 解析速度前馈（tau_ff=0）。

    返回本段最终的指令位置 q_goal，供调用方维持指令流连续。
    """
    n = arm.num_joints
    q_start = np.asarray(q_start, dtype=float).reshape(n)
    q_goal = np.asarray(q_goal, dtype=float).reshape(n)
    dq = q_goal - q_start
    dist = float(np.max(np.abs(dq)))
    if dist < 1e-6:
        arm.mit(pos=q_goal, vel=np.zeros(n), kp=kp, kd=kd, tau=np.zeros(n),
                request_feedback=True)
        return q_goal

    # 受最大关节速度约束自动延长时间，保证段内不超速
    duration = max(duration, dist / max(max_vel, 1e-3))
    dt = 1.0 / arm._rate
    steps = max(int(duration / dt), 1)

    for step in range(1, steps + 1):
        s = step / steps
        q_cmd = q_start + _quintic_alpha(s) * dq
        # 解析速度前馈：端点为 0，平滑且与循环实际频率无关
        v_cmd = _quintic_alpha_dot(s) * dq / duration
        v_norm = float(np.max(np.abs(v_cmd)))
        if v_norm > max_vel:
            v_cmd = v_cmd * (max_vel / v_norm)

        arm.mit(
            pos=q_cmd,
            vel=v_cmd,
            kp=kp,
            kd=kd,
            tau=np.zeros(n),
            request_feedback=True,
        )
        time.sleep(max(dt - 1e-5, 0.0))

    # 末点静止（速度归零），与后续静止保持指令连续
    arm.mit(pos=q_goal, vel=np.zeros(n), kp=kp, kd=kd, tau=np.zeros(n),
            request_feedback=True)
    return q_goal


def _reject_outliers_3sigma(samples: np.ndarray) -> np.ndarray:
    if samples.shape[0] < 4:
        return samples
    mu = np.mean(samples, axis=0)
    sigma = np.std(samples, axis=0)
    mask = np.all(np.abs(samples - mu) < 3.0 * (sigma + 1e-9), axis=1)
    if np.sum(mask) < 2:
        return samples
    return samples[mask]


def _sample_static(
    arm,
    q_hold: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    settle_s: float,
    sample_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在指令位置 q_hold 静止保持并采样。

    指令位置取上一段 move_j 的终点（而非重读实际值），保证指令流连续、
    避免段间阶跃；实际测量角 q 仍从反馈读取用于回归。
    """
    n = arm.num_joints
    q_hold = np.asarray(q_hold, dtype=float).reshape(n)
    t_end = time.perf_counter() + settle_s + sample_s
    settle_end = time.perf_counter() + settle_s
    qs, vs, taus = [], [], []

    while time.perf_counter() < t_end:
        arm.mit(
            pos=q_hold,
            vel=np.zeros(n),
            kp=kp,
            kd=kd,
            tau=np.zeros(n),
            request_feedback=True,
        )
        if time.perf_counter() >= settle_end:
            q, v, t = arm.get_state()
            qs.append(q)
            vs.append(v)
            taus.append(t)
        time.sleep(1.0 / arm._rate)

    qs_a = np.array(qs)
    taus_a = np.array(taus)
    qs_a = _reject_outliers_3sigma(qs_a)
    taus_a = _reject_outliers_3sigma(taus_a)
    return (
        np.mean(qs_a, axis=0),
        np.mean(taus_a, axis=0),
        np.std(taus_a, axis=0),
    )


def run_collection(
    arm,
    plan: PathPlan,
    *,
    resume_from: Optional[str] = None,
    on_progress=None,
) -> Tuple[List[TargetSample], Dict[str, dict]]:
    """执行路径采集。返回 (完成的 TargetSample 列表, 进行中缓存)。"""
    kp = plan.mit_kp
    kd = plan.mit_kd
    n = arm.num_joints

    pending_targets: Dict[str, Dict[str, dict]] = {}
    completed: List[TargetSample] = []
    skipping = resume_from is not None
    interrupted = {"flag": False}

    def _sigint(sig, frame):
        interrupted["flag"] = True

    old_handler = signal.signal(signal.SIGINT, _sigint)

    try:
        # 预热反馈并持续锁定当前位置，避免使能后无指令导致飞车
        q_curr = arm.get_positions(request=True)
        for _ in range(20):
            arm.mit(
                pos=q_curr,
                vel=np.zeros(arm.num_joints),
                kp=plan.mit_kp,
                kd=plan.mit_kd,
                tau=np.zeros(arm.num_joints),
                request_feedback=True,
            )
            time.sleep(1.0 / arm._rate)
        q_curr = arm.get_positions(request=True)

        for i, wp in enumerate(plan.waypoints):
            if interrupted["flag"]:
                break

            if skipping:
                was_skipping = True
                if wp.kind == "target" and wp.pose_name == resume_from:
                    skipping = False
                elif wp.pose_name != resume_from:
                    continue
                else:
                    skipping = False
                # 续采恢复点：从当前实际位置起步，避免大阶跃
                if not skipping and was_skipping:
                    q_curr = arm.get_positions(request=True)

            if on_progress:
                on_progress(i, len(plan.waypoints), wp)

            q_curr = move_j(
                arm,
                q_curr,
                wp.q,
                plan.move_duration_s,
                kp,
                kd,
                plan.max_velocity_rad_s,
                plan.max_acceleration_rad_s2,
            )

            if wp.kind != "target":
                continue

            # 静止保持锁定在本段指令终点 q_curr（= wp.q），保持指令流连续
            q_act, tau_mean, tau_std = _sample_static(
                arm,
                q_curr,
                kp,
                kd,
                plan.settle_time_s,
                plan.sample_time_s,
            )

            pname = wp.pose_name
            direction = wp.direction or "unknown"
            if pname not in pending_targets:
                pending_targets[pname] = {}

            pending_targets[pname][direction] = {
                "q_target": wp.q.copy(),
                "q_actual": q_act,
                "tau": tau_mean,
                "tau_std": tau_std,
            }

            if "minus" in pending_targets[pname] and "plus" in pending_targets[pname]:
                m = pending_targets[pname]["minus"]
                p = pending_targets[pname]["plus"]
                tau_static = 0.5 * (m["tau"] + p["tau"])
                q_actual = 0.5 * (m["q_actual"] + p["q_actual"])
                completed.append(
                    TargetSample(
                        pose_name=pname,
                        q_target=m["q_target"],
                        q_actual=q_actual,
                        tau_static=tau_static,
                        tau_plus=p["tau"],
                        tau_minus=m["tau"],
                        tau_std=0.5 * (m["tau_std"] + p["tau_std"]),
                        friction_delta=p["tau"] - m["tau"],
                    )
                )
                del pending_targets[pname]

    finally:
        signal.signal(signal.SIGINT, old_handler)

    return completed, pending_targets
