"""生成 80 个经限位+碰撞校验的标定种子位姿，写入 config/calibration_poses_80pt.yaml。

采样点数 = 位姿数 × 2（双向接近）= 160。
joint1 和 joint6 锁定为 0，仅 joint2-5 参与激励。
所有 target 及 target±approach_delta 均通过自碰撞与限位检查。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pinocchio as pin

from reBotArm_control_py.calibration.io import default_urdf_path, expand_arm_q
from reBotArm_control_py.calibration.pose_validator import build_pose_validator

APPROACH = 0.10
MARGIN = 0.05
SEED = 80801

FIX_J1 = 0.0
FIX_J6 = 0.0
N_BINS = 10
PER_BIN = 8
J2_RANGE = (0.10, 3.00)

LOWER = np.array([-2.8, 0.0, 0.0, -1.57, -1.57, -3.14])
UPPER = np.array([2.8, 3.14, 3.14, 1.57, 1.57, 3.14])


def fast_collision_ok(v, q6: np.ndarray) -> bool:
    q = expand_arm_q(q6, v.model.nq)
    pin.forwardKinematics(v.model, v.data, q)
    pin.updateGeometryPlacements(v.model, v.data, v.geom_model, v.geom_data, q)
    pin.computeCollisions(v.model, v.data, v.geom_model, v.geom_data, q, True)
    for k in range(len(v.geom_model.collisionPairs)):
        if v.geom_data.collisionResults[k].isCollision():
            return False
    return True


def limits_ok(q6: np.ndarray) -> bool:
    lo = LOWER + MARGIN
    hi = UPPER - MARGIN
    return bool(np.all(q6 >= lo - 1e-9) and np.all(q6 <= hi + 1e-9))


def pose_valid(v, q6: np.ndarray, delta: np.ndarray) -> bool:
    for sign in (0.0, 1.0, -1.0):
        qc = q6 + sign * delta
        if not limits_ok(qc):
            return False
        if not fast_collision_ok(v, qc):
            return False
    return True


def main() -> int:
    GROUND_CLEARANCE = 0.02  # 20mm 最小离地间距

    rng = np.random.default_rng(SEED)
    v = build_pose_validator(enable_collision=True, limit_margin=MARGIN,
                             enable_ground=True, ground_z=GROUND_CLEARANCE)
    if not v.collision_enabled:
        raise RuntimeError("碰撞检测不可用，无法生成安全位姿")
    print(f"碰撞对数量(含地面): {len(v.geom_model.collisionPairs)}")

    # joint1, joint6 锁定 → approach 偏移对应分量为 0
    delta = np.array([0.0, APPROACH, APPROACH, APPROACH, APPROACH, 0.0])

    edges = np.linspace(J2_RANGE[0], J2_RANGE[1], N_BINS + 1)
    poses: list[np.ndarray] = []
    for bi in range(N_BINS):
        lo2, hi2 = edges[bi], edges[bi + 1]
        got, tries = 0, 0
        while got < PER_BIN and tries < 80000:
            tries += 1
            q6 = np.array([
                FIX_J1,
                rng.uniform(lo2, hi2),
                rng.uniform(0.10, 3.00),
                rng.uniform(-1.40, 1.40),
                rng.uniform(-1.40, 1.40),
                FIX_J6,
            ])
            if any(np.linalg.norm(q6 - p) < 0.35 for p in poses):
                continue
            if pose_valid(v, q6, delta):
                poses.append(q6)
                got += 1
        print(f"  档 {bi+1:2d} [{np.degrees(lo2):5.0f}-{np.degrees(hi2):5.0f}°]: "
              f"采纳 {got}/{PER_BIN} (试 {tries})")

    print(f"总采纳 {len(poses)} 个位姿（joint1={FIX_J1}, joint6={FIX_J6}）")

    # 交错排序：将 10 个档按"低-高"配对交替排列，避免连续停留高角度区域。
    # 配对顺序: bin0-bin5, bin1-bin6, bin2-bin7, bin3-bin8, bin4-bin9
    # 每对内再按 low→high→low→high... 轮流取，使 j2 角度持续在低高之间跳跃。
    bins_poses: list[list[np.ndarray]] = [[] for _ in range(N_BINS)]
    for p in poses:
        bi = min(int((p[1] - J2_RANGE[0]) / (J2_RANGE[1] - J2_RANGE[0]) * N_BINS), N_BINS - 1)
        bins_poses[bi].append(p)

    half = N_BINS // 2
    interleaved: list[np.ndarray] = []
    for slot in range(PER_BIN):
        for pair_idx in range(half):
            lo_bin = pair_idx
            hi_bin = pair_idx + half
            if slot < len(bins_poses[lo_bin]):
                interleaved.append(bins_poses[lo_bin][slot])
            if slot < len(bins_poses[hi_bin]):
                interleaved.append(bins_poses[hi_bin][slot])
    poses = interleaved

    # 统计连续高角度情况
    j2_seq = [np.degrees(p[1]) for p in poses]
    max_consec_high = 0
    cur = 0
    for d in j2_seq:
        if d > 150:
            cur += 1
            max_consec_high = max(max_consec_high, cur)
        else:
            cur = 0
    print(f"[排序] 交错后最大连续 j2>150° 数: {max_consec_high}")

    home = np.array([FIX_J1, 1.0, 1.2, 0.0, 0.0, FIX_J6])
    if not pose_valid(v, home, delta):
        home = poses[0]

    lines: list[str] = []
    lines.append("# 重力补偿标定 — 80 点种子位姿（程序 1 输入）")
    lines.append("# 单位: rad。approach_delta: 双向接近中转偏移，用于程序 2 消 Coulomb 摩擦。")
    lines.append("# 由 scripts/gen_calib_poses_80.py 自动生成：joint1/joint6 锁定为 0；")
    lines.append("# joint2 角度均匀分 10 档 × 8 个/档 = 80 位姿；")
    lines.append("# 低角度档与高角度档交错排列，避免连续停留在 j2>150° 高力矩区域。")
    lines.append("# 所有位姿及双向接近点均通过限位+自碰撞+地面干涉检查。")
    lines.append(f"# 位姿数 {len(poses)} → 静止采样点约 {len(poses) * 2} 个。")
    lines.append("")
    lines.append("urdf: null  # null = 内置 00-arm-rs_asm-v3.urdf")
    lines.append("")
    lines.append("home_q: [" + ", ".join(f"{x:.4f}" for x in home) + "]")
    lines.append("")
    lines.append("# 路径规划参数")
    lines.append("segment_samples: 24")
    lines.append("max_segment_dq: 0.35")
    lines.append(f"limit_margin: {MARGIN}")
    lines.append("")
    lines.append("# 数据采集参数（程序 2 读取 path 文件中的副本）")
    lines.append("settle_time_s: 0.5")
    lines.append("sample_time_s: 1.0")
    lines.append("move_duration_s: 2.0")
    lines.append("max_velocity_rad_s: 0.5")
    lines.append("max_acceleration_rad_s2: 1.5")
    lines.append("")
    lines.append("# 程序 2 MIT 锁位增益（高刚度便于读静态力矩）")
    lines.append("mit_kp: [80.0, 80.0, 80.0, 15.0, 15.0, 15.0]")
    lines.append("mit_kd: [4.0, 4.0, 4.0, 1.5, 1.5, 1.5]")
    lines.append("")
    lines.append("poses:")
    for i, q6 in enumerate(poses, 1):
        qs = ", ".join(f"{x:.4f}" for x in q6)
        ds = ", ".join(f"{d:.2f}" for d in delta)
        lines.append(f"  - name: pose_{i:02d}")
        lines.append(f"    q: [{qs}]")
        lines.append(f"    approach_delta: [{ds}]")
        lines.append("")

    out = Path(default_urdf_path()).resolve().parents[3] / "config" / "calibration_poses_80pt.yaml"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写入 {out}")
    j2deg = np.degrees([p[1] for p in poses])
    print(f"[分布] joint2 0-90°: {int(np.sum(j2deg <= 90))} 点  "
          f">90°: {int(np.sum(j2deg > 90))} 点")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
