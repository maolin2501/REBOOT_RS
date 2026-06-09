#!/usr/bin/env python3
"""程序 16：手柄遥操作（增量式笛卡尔空间末端控制）。

通过 Linux joystick API 直接读取手柄输入，结合 reBotArm 的
``RobotArm`` 200/100Hz MIT 控制、Pinocchio 正逆运动学与标定重力补偿，
实现末端坐标空间的实时遥操作。

控制映射:
  左摇杆 Y/X     → 末端 X/Y 平移
  LT / RT         → 末端 Z 下/上
  右摇杆 Y/X     → Roll / Yaw 旋转
  LB / RB         → Pitch 旋转
  A               → 切换速度档位（5 档循环）
  B               → 回 Home 位置 [0, 1.0, 1.2, 0, 0, 0]
  X               → 回零位 [0]×6
  Y               → 切换零力矩模式（可手动拖动）
  D-pad 上/下     → 夹爪 开/合
  Back            → 急停 / 解除（位置锁定）
  Start           → 退出程序

使用前确保:
  1. CAN 接口已激活: sudo ip link set can0 up type can bitrate 1000000
  2. 机械臂已上电、已标定零点（example/2_zero_and_read.py）
  3. 手柄已连接 (USB 或蓝牙): ls /dev/input/js*
  4. 已安装依赖: pin(pinocchio)、motorbridge、numpy、pyyaml

用法:
  python example/16_xbox_teleop.py                          # 自动识别 profile
  python example/16_xbox_teleop.py --js /dev/input/js1     # 指定手柄设备
  python example/16_xbox_teleop.py --profile xbox_default  # 强制指定映射
  python example/16_xbox_teleop.py --list-profiles          # 查看内置映射
  python example/16_xbox_teleop.py --dump-input             # 仅打印手柄原始输入
  python example/16_xbox_teleop.py --no-gripper             # 不连接夹爪
  python example/16_xbox_teleop.py --max-lin-vel 0.10       # 限制最大线速度
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reBotArm_control_py.actuator import RobotArm, Gripper
from reBotArm_control_py.teleop import LinuxJoystick, XboxTeleopController
from reBotArm_control_py.teleop.controller_profiles import (
    PROFILES,
    ControllerProfile,
    detect_controller,
)

logger = logging.getLogger("xbox_teleop")


def dump_input(joy: LinuxJoystick, profile: ControllerProfile) -> None:
    """仅打印手柄原始输入，用于调试轴/按钮索引与映射。"""
    print("\n" + "=" * 60)
    print("  手柄输入调试模式")
    print("=" * 60)
    print(f"设备: {joy.device}")
    print(f"Profile: {profile.display_name} [{profile.profile_id}]")
    print("按 Ctrl+C 退出，移动摇杆或按键查看原始索引变化。")

    last_axes = [None] * len(joy.axes)
    last_buttons = [None] * len(joy.buttons)
    try:
        while joy.connected:
            changed = False
            for idx, value in enumerate(joy.axes):
                prev = last_axes[idx]
                if prev is None or abs(value - prev) >= 0.05:
                    print(f"axis[{idx}] = {value:+.3f}")
                    last_axes[idx] = value
                    changed = True
            for idx, value in enumerate(joy.buttons):
                prev = last_buttons[idx]
                if prev is None or value != prev:
                    print(f"button[{idx}] = {value}")
                    last_buttons[idx] = value
                    changed = True
            if not changed:
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="reBotArm 手柄遥操作（增量笛卡尔控制）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s
  %(prog)s --js /dev/input/js1 --profile xbox_default
  %(prog)s --max-lin-vel 0.10 --max-ang-vel 1.0
  %(prog)s --dump-input
  %(prog)s --list-profiles
""",
    )
    parser.add_argument("--js", default="/dev/input/js0",
                        help="手柄设备路径 (默认: /dev/input/js0)")
    parser.add_argument("--profile", default="auto",
                        choices=["auto", *PROFILES.keys()],
                        help="控制器映射 profile (默认: auto)")
    parser.add_argument("--list-profiles", action="store_true",
                        help="列出内置 profile 并退出")
    parser.add_argument("--dump-input", action="store_true",
                        help="仅打印手柄原始输入并退出")
    parser.add_argument("--rate", type=float, default=100.0,
                        help="输入处理频率 Hz (默认: 100)")
    parser.add_argument("--max-lin-vel", type=float, default=0.15,
                        help="最大线速度 m/s (默认: 0.15)")
    parser.add_argument("--max-ang-vel", type=float, default=1.5,
                        help="最大角速度 rad/s (默认: 1.5)")
    parser.add_argument("--deadzone", type=float, default=None,
                        help="摇杆死区 (默认: 使用 profile 推荐值)")
    parser.add_argument("--tcp-offset", type=float, default=0.15,
                        help="末端跟踪点距 joint6 前向(+Z)偏移 m (默认: 0.15)")
    parser.add_argument("--zt-kd", type=float, default=0.5,
                        help="零力矩模式阻尼 kd (默认: 0.5，越大越粘滞)")
    parser.add_argument("--no-gravity-ff", action="store_true",
                        help="关闭重力前馈补偿")
    parser.add_argument("--no-gripper", action="store_true",
                        help="不连接夹爪")
    parser.add_argument("--debug", action="store_true",
                        help="调试日志模式")
    args = parser.parse_args()

    if args.list_profiles:
        for profile_id, profile in PROFILES.items():
            print(f"{profile_id}: {profile.display_name} - {profile.description}")
        return 0

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(name)s][%(levelname)s] %(message)s",
    )

    detection = detect_controller(args.js, requested_profile=args.profile)
    deadzone = (args.deadzone if args.deadzone is not None
                else detection.profile.default_deadzone)
    logger.info(
        "手柄检测: device=%s name=%s vid=%s pid=%s profile=%s source=%s",
        detection.resolved_device,
        detection.name or "unknown",
        detection.vendor or "unknown",
        detection.product or "unknown",
        detection.profile.profile_id,
        detection.source,
    )

    # ---- 连接手柄 ----
    joy = LinuxJoystick(device=args.js)
    if not joy.connect():
        print(f"\n无法打开手柄 {args.js}")
        print("请确认:")
        print("  1. 手柄已连接 (蓝牙或 USB)")
        print("  2. 设备存在: ls /dev/input/js*")
        print("  3. 权限足够: sudo chmod 666 /dev/input/js0")
        print("  4. 驱动已加载: sudo modprobe joydev")
        return 1
    logger.info("手柄已连接: %s", args.js)

    if args.dump_input:
        try:
            dump_input(joy, detection.profile)
        finally:
            joy.disconnect()
        return 0

    # ---- 连接机械臂 ----
    try:
        arm = RobotArm()
    except Exception as e:
        joy.disconnect()
        print(f"\n机械臂连接失败: {e}")
        print("请确认 CAN 接口已激活:")
        print("  sudo ip link set can0 up type can bitrate 1000000")
        return 1

    arm.connect()
    arm.mode_mit()
    arm.enable()
    logger.info("机械臂已使能 (MIT 模式)")

    # ---- 连接夹爪（可选）----
    gripper = None
    if not args.no_gripper:
        try:
            gripper = Gripper()
            gripper.connect()
            gripper.mode_pos_vel()
            if gripper.enable():
                logger.info("夹爪已使能 (POS_VEL 模式)")
            else:
                logger.warning("夹爪使能失败，禁用夹爪功能")
                gripper.disconnect()
                gripper = None
        except Exception as e:
            logger.warning("夹爪连接失败（将仅控制机械臂）: %s", e)
            gripper = None

    # ---- 信号处理 ----
    shutdown = threading.Event()

    def on_signal(_sig, _frame):
        shutdown.set()
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    controller = XboxTeleopController(
        arm=arm,
        joystick=joy,
        profile=detection.profile,
        gripper=gripper,
        update_rate=args.rate,
        max_linear_velocity=args.max_lin_vel,
        max_angular_velocity=args.max_ang_vel,
        deadzone=deadzone,
        tcp_offset=args.tcp_offset,
        gravity_ff=not args.no_gravity_ff,
        zero_torque_kd=args.zt_kd,
    )

    controller.start()

    try:
        while not shutdown.is_set() and not controller.exit_requested:
            shutdown.wait(timeout=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n正在清理...")
        controller.stop()
        try:
            arm.disconnect()
        except Exception as e:
            logger.error("机械臂断开异常: %s", e)
        if gripper is not None:
            try:
                gripper.disconnect()
            except Exception as e:
                logger.error("夹爪断开异常: %s", e)
        joy.disconnect()
        print("已退出（机械臂已失能，请注意支撑）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
