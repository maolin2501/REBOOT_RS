#!/usr/bin/env python3
"""程序 4：标定后重力补偿验证（A/B 对比）。

MIT 重力前馈：tau = g(q)，kp/kd 可调。
按键: Ctrl+C 退出；输入 c 切换 URDF 名义 / 标定惯量。

用法:
    uv run python example/14_gravity_compensation_calibrated.py
"""
from __future__ import annotations

import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reBotArm_control_py.actuator import RobotArm
from reBotArm_control_py.calibration import default_config_dir
from reBotArm_control_py.calibration.io import expand_arm_q
from reBotArm_control_py.dynamics import (
    compute_generalized_gravity,
    load_dynamics_model,
)

_running = True
_use_calibrated = True
_counter = 0


def _sigint_handler(signum, frame):
    global _running
    print("\n[gravity_comp] 收到 Ctrl+C，准备停止...")
    _running = False


def _try_read_key() -> str | None:
    if not sys.stdin.isatty():
        return None
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def main() -> None:
    global _use_calibrated, _counter

    cal_yaml = default_config_dir() / "inertia_calibrated.yaml"
    if not cal_yaml.is_file():
        print(f"[错误] 未找到标定文件: {cal_yaml}")
        print("  请先运行程序 3: example/13_calib_solve.py")
        sys.exit(1)

    import signal

    signal.signal(signal.SIGINT, _sigint_handler)

    model_urdf = load_dynamics_model()
    model_cal = load_dynamics_model(calibration_yaml=str(cal_yaml))

    print("=" * 60)
    print("  重力补偿验证（标定 A/B）")
    print(f"  标定文件: {cal_yaml}")
    print("  当前模式: 标定惯量")
    print("  输入 c 切换 URDF/标定；Ctrl+C 退出")
    print("=" * 60)

    arm = RobotArm()
    arm.connect()
    arm.enable()
    arm.mode_mit(kp=np.full(arm.num_joints, 2.0), kd=np.full(arm.num_joints, 1.0))

    old_settings = None
    if sys.stdin.isatty():
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    def controller(arm_ref: RobotArm, dt: float) -> None:
        global _use_calibrated, _counter
        key = _try_read_key()
        if key == "c":
            _use_calibrated = not _use_calibrated
            mode = "标定" if _use_calibrated else "URDF"
            print(f"\n[切换] 重力模型 → {mode}")

        q_arm = arm_ref.get_positions()
        model = model_cal if _use_calibrated else model_urdf
        q_full = expand_arm_q(q_arm, model.nq)
        tau_g_full = compute_generalized_gravity(model=model, q=q_full)
        tau_g = tau_g_full[: arm_ref.num_joints]

        model_b = model_urdf if _use_calibrated else model_cal
        tau_other_full = compute_generalized_gravity(model=model_b, q=q_full)
        tau_other = tau_other_full[: arm_ref.num_joints]

        arm_ref.mit(
            pos=q_arm,
            vel=np.zeros(arm_ref.num_joints),
            kp=np.full(arm_ref.num_joints, 2.0),
            kd=np.full(arm_ref.num_joints, 1.0),
            tau=tau_g,
            request_feedback=True,
        )

        _counter += 1
        if _counter % 25 == 0:
            mode = "CAL" if _use_calibrated else "URDF"
            diff = tau_g - tau_other
            print(
                f"[{_counter:4d}] {mode}  tau_g="
                + " ".join(f"{t:+.3f}" for t in tau_g)
                + f"  Δ={np.linalg.norm(diff):.3f} N·m"
            )

    arm.start_control_loop(controller, rate=arm._rate)
    print(f"[控制循环] @ {arm._rate} Hz")

    try:
        while _running:
            time.sleep(0.05)
    finally:
        if old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        arm.disconnect()
        print("[完成] 已断开")


if __name__ == "__main__":
    main()
