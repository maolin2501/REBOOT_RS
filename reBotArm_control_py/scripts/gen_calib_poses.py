"""生成约 40 个经限位+碰撞校验的标定种子位姿，写入 config/calibration_poses.yaml。

采样点数 = 位姿数 × 2（双向接近）≈ 80。
所有 target 及 target±approach_delta 均通过自碰撞与限位检查。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin

from reBotArm_control_py.calibration.io import default_urdf_path, expand_arm_q
from reBotArm_control_py.calibration.pose_validator import build_pose_validator

APPROACH = 0.10
MARGIN = 0.05
SEED = 12345

# 方案C：joint1 固定不动(不参与标定/不移动)，joint2 角度均匀分档覆盖 0-175°
FIX_J1 = 0.0
N_BINS = 10               # joint2 角度分档数
PER_BIN = 4               # 每档位姿数 → 总数 = N_BINS*PER_BIN
J2_RANGE = (0.10, 3.00)   # joint2 采样范围(rad)，约 6-172°

# 新 URDF 限位（rad）
LOWER = np.array([-2.8, 0.0, 0.0, -1.57, -1.57, -3.14])
UPPER = np.array([2.8, 3.14, 3.14, 1.57, 1.57, 3.14])


def fast_collision_ok(v, q6: np.ndarray) -> bool:
    """仅做布尔自碰撞判断（跳过昂贵的 computeDistances）。"""
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
    """target 及双向接近点全部合法。"""
    for sign in (0.0, 1.0, -1.0):
        qc = q6 + sign * delta
        if not limits_ok(qc):
            return False
        if not fast_collision_ok(v, qc):
            return False
    return True


def main() -> int:
    rng = np.random.default_rng(SEED)
    v = build_pose_validator(enable_collision=True, limit_margin=MARGIN,
                             enable_ground=True)
    if not v.collision_enabled:
        raise RuntimeError("碰撞检测不可用，无法生成安全位姿")
    print(f"碰撞对数量(含地面): {len(v.geom_model.collisionPairs)}")

    # joint1 固定不动 → approach 偏移 joint1 分量为 0
    delta = np.array([0.0, APPROACH, APPROACH, APPROACH, APPROACH, APPROACH])

    # joint2 分档均匀采样，保证 0-90° 与 >90° 覆盖均衡；其余关节随机激励
    edges = np.linspace(J2_RANGE[0], J2_RANGE[1], N_BINS + 1)
    poses: list[np.ndarray] = []
    for bi in range(N_BINS):
        lo2, hi2 = edges[bi], edges[bi + 1]
        got, tries = 0, 0
        while got < PER_BIN and tries < 40000:
            tries += 1
            q6 = np.array([
                FIX_J1,                       # joint1 固定不动
                rng.uniform(lo2, hi2),        # joint2 在本档
                rng.uniform(0.10, 3.00),      # joint3
                rng.uniform(-1.40, 1.40),     # joint4
                rng.uniform(-1.40, 1.40),     # joint5
                rng.uniform(-3.00, 3.00),     # joint6
            ])
            if any(np.linalg.norm(q6 - p) < 0.45 for p in poses):
                continue
            if pose_valid(v, q6, delta):
                poses.append(q6)
                got += 1
        print(f"  档 {bi+1:2d} [{np.degrees(lo2):5.0f}-{np.degrees(hi2):5.0f}°]: 采纳 {got}/{PER_BIN} (试 {tries})")

    print(f"总采纳 {len(poses)} 个位姿（joint1 固定 {FIX_J1}）")

    # home: joint1 固定的安全肩构型
    home = np.array([FIX_J1, 1.0, 1.2, 0.0, 0.0, 0.0])
    if not pose_valid(v, home, delta):
        home = poses[0]

    lines: list[str] = []
    lines.append("# 重力补偿标定 — 种子位姿（程序 1 输入）")
    lines.append("# 单位: rad。approach_delta: 双向接近中转偏移，用于程序 2 消 Coulomb 摩擦。")
    lines.append("# 由 scripts/gen_calib_poses.py 自动生成：joint1 固定不动；joint2 角度均匀分档；")
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

    out = Path(default_urdf_path()).resolve().parents[3] / "config" / "calibration_poses.yaml"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写入 {out}")
    j2deg = np.degrees([p[1] for p in poses])
    print(f"[分布] joint2 0-90°: {int(np.sum(j2deg <= 90))} 点  >90°: {int(np.sum(j2deg > 90))} 点")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
