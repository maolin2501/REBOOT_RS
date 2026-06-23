#!/usr/bin/env python3
"""程序 2：标定数据采集（在线）。

沿 calibration_path.yaml 逐 waypoint 运动，在 target 点双向静止采样力矩。

用法:
    uv run python example/12_calib_collect.py
    uv run python example/12_calib_collect.py --resume pose_09_j4
    uv run python example/12_calib_collect.py --dry-run
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reBotArm_control_py.calibration import (
    default_config_dir,
    default_urdf_path,
    file_sha256,
    load_path,
    run_collection,
    save_measurements,
)
from reBotArm_control_py.calibration.io import expand_arm_q
from reBotArm_control_py.calibration.pose_validator import build_pose_validator


def main() -> int:
    parser = argparse.ArgumentParser(description="重力补偿标定 — 程序2 数据采集")
    parser.add_argument(
        "--path",
        type=str,
        default=str(default_config_dir() / "calibration_path.yaml"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(default_config_dir() / "calibration_measurements.npz"),
    )
    parser.add_argument("--resume", type=str, default=None, help="从该 pose_name 续采")
    parser.add_argument("--dry-run", action="store_true", help="仅打印路径，不连接机器人")
    args = parser.parse_args()

    plan = load_path(args.path)
    urdf_path = plan.urdf_path
    urdf_sha = file_sha256(urdf_path)
    if urdf_sha != plan.urdf_sha256:
        print("[错误] URDF 哈希与路径文件不一致，请重新运行程序 1")
        print(f"  文件记录: {plan.urdf_sha256[:16]}...")
        print(f"  当前 URDF: {urdf_sha[:16]}...")
        return 1

    n_targets = sum(1 for w in plan.waypoints if w.kind == "target")
    print("=" * 60)
    print("  程序 2：标定数据采集")
    print(f"  路径: {args.path}")
    print(f"  waypoint: {len(plan.waypoints)}  target: {n_targets}")
    print(f"  URDF SHA256: {urdf_sha[:16]}...")
    print("=" * 60)

    if args.dry_run:
        for i, wp in enumerate(plan.waypoints):
            print(
                f"  [{i:3d}] {wp.kind:16s} {wp.pose_name:20s} "
                f"q={np.rad2deg(wp.q).round(1)}"
            )
        return 0

    try:
        from reBotArm_control_py.actuator import RobotArm
    except ImportError as exc:
        print(f"[错误] 需要 motorbridge: {exc}")
        return 1

    validator = build_pose_validator(
        urdf_path=urdf_path,
        limit_margin=plan.limit_margin,
        enable_collision=False,
    )

    arm = RobotArm()
    arm.connect()
    records_cache: list = []
    interrupted = False

    def _on_progress(i, total, wp):
        print(
            f"\r  [{i+1}/{total}] {wp.kind:14s} {wp.name:28s}",
            end="",
            flush=True,
        )

    def _sigint(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\n[中断] 正在安全停止...")

    signal.signal(signal.SIGINT, _sigint)

    try:
        # 使能前先读取当前位置，用于使能后立即锁定
        for _ in range(10):
            arm._request_and_poll()
            time.sleep(0.02)
        q_now = arm.get_positions(request=True)

        arm.enable()
        arm.mode_mit(kp=plan.mit_kp, kd=plan.mit_kd)

        # 使能+切模式后立即发送"锁住当前位置"指令，消除飞车窗口
        for _ in range(20):
            arm.mit(
                pos=q_now,
                vel=np.zeros(arm.num_joints),
                kp=plan.mit_kp,
                kd=plan.mit_kd,
                tau=np.zeros(arm.num_joints),
                request_feedback=True,
            )
            time.sleep(1.0 / arm._rate)
        q_now = arm.get_positions(request=True)
        print(f"[使能] OK  (锁定位置: {np.rad2deg(q_now).round(1)}°)")

        # 运动前限位复检
        for wp in plan.waypoints:
            ok, reason = validator.check_limits(wp.q)
            if not ok:
                print(f"[错误] waypoint {wp.name} 越限: {reason}")
                return 1

        records, pending = run_collection(
            arm,
            plan,
            resume_from=args.resume,
            on_progress=_on_progress,
        )
        records_cache = records
        print(f"\n[采集] 完成 {len(records)} 个 target 位姿")
        if pending:
            print(f"[警告] 未完成配对: {list(pending.keys())}")

    except Exception as exc:
        print(f"\n[异常] {exc}")
        interrupted = True
    finally:
        try:
            arm.disable()
        except Exception:
            pass
        arm.disconnect()
        print("[断开] OK")

    if records_cache:
        meta = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "urdf_path": urdf_path,
            "urdf_sha256": urdf_sha,
            "path_file": str(args.path),
            "num_records": len(records_cache),
            "resume_from": args.resume,
            "interrupted": interrupted,
        }
        save_measurements(args.output, records_cache, meta)
        print(f"\n[保存] {args.output} ({len(records_cache)} 条记录)")
        print("\n[摘要] pose          |tau_static|  friction_delta")
        print("-" * 55)
        for r in records_cache:
            print(
                f"  {r.pose_name:22s} "
                f"{np.linalg.norm(r.tau_static):7.3f} N·m  "
                f"{np.linalg.norm(r.friction_delta):7.3f} N·m"
            )
    elif interrupted:
        print("[提示] 无已保存数据，请使用 --resume 续采")
        return 130

    return 0 if not interrupted else 130


if __name__ == "__main__":
    raise SystemExit(main())
