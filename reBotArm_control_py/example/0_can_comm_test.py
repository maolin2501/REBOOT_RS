#!/usr/bin/env python3
"""逐电机 CAN 通信测试 (不使能) — 基于 SocketCAN can0。

按 config/arm.yaml 注册全部电机，但**不调用任何使能/控制指令**，
仅对每个电机做通信探测：
    1) robstride_ping_host_id  — 探测电机是否在总线上应答 (RobStride)
    2) request_feedback + poll + get_state — 读取一帧反馈数据

电机始终保持失能，全程不会有任何运动。

用法:
    python example/0_can_comm_test.py [配置路径]
默认配置: config/arm.yaml (channel=can0)

前置条件 (需要 root，一次性):
    sudo ip link set can0 type can bitrate 1000000
    sudo ip link set can0 up
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reBotArm_control_py.actuator.arm import load_cfg
from motorbridge import Controller, CallError


def make_controller(channel: str) -> Controller:
    if channel.startswith("/dev/tty"):
        return Controller.from_dm_serial(channel, 921600)
    return Controller(channel)


def add_motor(ctrl: Controller, jc):
    if jc.vendor == "damiao":
        return ctrl.add_damiao_motor(jc.motor_id, jc.feedback_id, jc.model)
    if jc.vendor == "myactuator":
        return ctrl.add_myactuator_motor(jc.motor_id, jc.feedback_id, jc.model)
    if jc.vendor == "robstride":
        return ctrl.add_robstride_motor(jc.motor_id, jc.feedback_id, jc.model)
    if jc.vendor == "hightorque":
        return ctrl.add_hightorque_motor(jc.motor_id, jc.feedback_id, jc.model)
    raise ValueError(f"Unsupported vendor: {jc.vendor}")


def test_ping(mot, jc):
    """RobStride 专用：ping 探测电机应答。返回 (ok, info)。"""
    if jc.vendor != "robstride":
        return None, "skip (非 robstride)"
    try:
        dev_id, resp_id = mot.robstride_ping_host_id(jc.feedback_id, 500)
        return True, f"device_id={dev_id:#04x} responder_id={resp_id:#04x}"
    except CallError as e:
        return False, str(e)


def test_feedback(ctrl, mot, retries=10, settle=0.01):
    """通用：请求反馈并读取一帧状态。返回 (ok, state_or_msg)。"""
    st = None
    for _ in range(retries):
        try:
            mot.request_feedback()
            ctrl.poll_feedback_once()
        except CallError as e:
            return False, str(e)
        time.sleep(settle)
        try:
            st = mot.get_state()
        except CallError as e:
            return False, str(e)
        if st is not None:
            return True, st
    return False, "无反馈数据 (超时)"


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).resolve().parents[1] / "config" / "arm.yaml"
    )
    cfg = load_cfg(cfg_path)
    channel = cfg["channel"]
    joints = cfg["joints"]

    print(f"配置文件 : {cfg_path}")
    print(f"通信通道 : {channel}")
    print(f"电机数量 : {len(joints)}")
    print("=" * 72)
    print("注意: 全程不使能、不发送控制指令，电机不会运动。\n")

    try:
        ctrl = make_controller(channel)
    except CallError as e:
        print(f"[致命] 无法打开通道 {channel}: {e}")
        print("请确认 can0 已配置并 up:")
        print("  sudo ip link set can0 type can bitrate 1000000")
        print("  sudo ip link set can0 up")
        sys.exit(1)

    results = []
    try:
        motors = {}
        for jc in joints:
            motors[jc.name] = add_motor(ctrl, jc)

        for jc in joints:
            mot = motors[jc.name]
            header = (f"[{jc.name}] id={jc.motor_id:#04x} "
                      f"host={jc.feedback_id:#04x} model={jc.model} "
                      f"vendor={jc.vendor}")
            print(header)

            ping_ok, ping_info = test_ping(mot, jc)
            if ping_ok is None:
                print(f"    ping     : {ping_info}")
            elif ping_ok:
                print(f"    ping     : OK   {ping_info}")
            else:
                print(f"    ping     : FAIL {ping_info}")

            fb_ok, fb = test_feedback(ctrl, mot)
            if fb_ok:
                deg = fb.pos * 180.0 / 3.14159265358979
                print(f"    feedback : OK   pos={deg:+.3f}deg "
                      f"vel={fb.vel:+.3f} torq={fb.torq:+.3f} "
                      f"status={fb.status_code} t_rotor={fb.t_rotor:.1f}")
            else:
                print(f"    feedback : FAIL {fb}")

            online = bool(ping_ok) or fb_ok
            results.append((jc.name, online))
            print()

    finally:
        try:
            ctrl.shutdown()
        except Exception:
            pass
        try:
            ctrl.close()
        except Exception:
            pass

    print("=" * 72)
    print("汇总:")
    ok_n = sum(1 for _, o in results if o)
    for name, online in results:
        print(f"    {name:8s} : {'在线 ✓' if online else '离线 ✗'}")
    print(f"\n在线 {ok_n}/{len(results)} 个电机")
    sys.exit(0 if ok_n == len(results) else 2)


if __name__ == "__main__":
    main()
