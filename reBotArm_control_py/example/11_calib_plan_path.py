#!/usr/bin/env python3
"""程序 1：标定路径规划（离线）。

读取种子位姿 → URDF 限位/自碰撞检验 → 生成无碰撞路径 YAML。

用法:
    uv run python example/11_calib_plan_path.py
    uv run python example/11_calib_plan_path.py --poses config/calibration_poses.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reBotArm_control_py.calibration import (
    build_pose_validator,
    default_config_dir,
    default_urdf_path,
    file_sha256,
    load_seed_config,
    plan_calibration_path,
    save_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="重力补偿标定 — 程序1 路径规划")
    parser.add_argument(
        "--poses",
        type=str,
        default=str(default_config_dir() / "calibration_poses.yaml"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(default_config_dir() / "calibration_path.yaml"),
    )
    parser.add_argument("--no-collision", action="store_true", help="仅关节限位，不做碰撞检测")
    parser.add_argument("--allow-partial", action="store_true",
                        help="部分位姿被拒时，仅用采纳位姿写入路径（不中断）")
    parser.add_argument("--ground-z", type=float, default=0.0,
                        help="地面碰撞面高度(m)，设>0可强制最小离地间距")
    args = parser.parse_args()

    cfg = load_seed_config(args.poses)
    urdf_path = cfg.urdf_path or str(default_urdf_path())
    urdf_sha = file_sha256(urdf_path)

    print("=" * 60)
    print("  程序 1：标定路径规划")
    print(f"  URDF: {urdf_path}")
    print(f"  SHA256: {urdf_sha[:16]}...")
    print("=" * 60)

    validator = build_pose_validator(
        urdf_path=urdf_path,
        limit_margin=cfg.limit_margin,
        enable_collision=not args.no_collision,
        enable_ground=not args.no_collision,
        ground_z=args.ground_z,
    )
    has_ground = any(
        validator.geom_model.geometryObjects[p.first].name == "ground_plane"
        or validator.geom_model.geometryObjects[p.second].name == "ground_plane"
        for p in validator.geom_model.collisionPairs
    ) if validator.collision_enabled else False
    print(f"[碰撞检测] {'启用' if validator.collision_enabled else '禁用（仅限位）'}"
          f"{'  + 水平面干涉检查' if has_ground else ''}")

    plan = plan_calibration_path(cfg, validator, urdf_path, urdf_sha)

    print(f"\n[路径] waypoint 数量: {len(plan.waypoints)}")
    print(f"[路径] 累计 |Δq| 之和: {plan.total_path_dq:.3f} rad")
    print(f"[路径] 最小净空: {plan.min_clearance_m:.4f} m")

    print("\n[种子位姿审查]")
    n_ok = sum(1 for r in plan.pose_reports if r.accepted)
    for r in plan.pose_reports:
        status = "采纳" if r.accepted else "拒绝"
        extra = f" — {r.reason}" if r.reason else ""
        print(f"  {r.name:24s} {status}{extra}")
    print(f"  合计: {n_ok}/{len(plan.pose_reports)} 采纳")

    all_ok = all(r.accepted for r in plan.pose_reports if r.name != "return_home")
    if not all_ok and not args.allow_partial:
        print("\n[失败] 存在未通过检验的位姿，未写入完整路径。"
              "（加 --allow-partial 可仅写采纳部分）")
        return 1
    if not all_ok:
        n_rej = sum(1 for r in plan.pose_reports
                    if not r.accepted and r.name != "return_home")
        print(f"\n[部分采纳] 跳过 {n_rej} 个失败位姿，仅写入采纳路径。")

    save_path(args.output, plan)
    print(f"\n[完成] 已写入: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
