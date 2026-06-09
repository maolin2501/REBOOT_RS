#!/usr/bin/env python3
"""零力矩 / 自由拖动模式（单文件汇总版，仅含重力补偿执行）。

把零力矩重力补偿的执行逻辑收进一个文件，不含任何标定采集/求解过程：
  - 加载 URDF 构建 6-DOF 臂模型（锁定夹爪关节）
  - 读取已标定的惯量 YAML（config/inertia_calibrated.yaml）写入模型
  - 控制律：MIT 前馈，kp=0，tau = g(q) + 零位偏置，叠加阻尼 kd
  - 可选自适应 kd：关节速度低时提高阻尼，拖动速度快时降低阻尼
  - 启动时把重力前馈从 0 缓升到满量（--ramp 秒），避免上电瞬间冲击
  - 输出端扭矩逆变换，抵消电机增益 a 与偏置 b

机械臂在任意姿态下"失重"，可用手自由拖动，松手后停在原地（不坠落）。
依赖唯一的外部硬件接口是 ``RobotArm``（CAN 电机驱动）。

用法:
    python example/zero_torque_mode_standalone.py
    python example/zero_torque_mode_standalone.py --kd 0.4 --ramp 1.5
    python example/zero_torque_mode_standalone.py --model urdf
    python example/zero_torque_mode_standalone.py --adaptive-kd

Ctrl+C 退出（退出前自动失能）。
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pinocchio as pin
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reBotArm_control_py.actuator import RobotArm

_running = True

# 标定连杆（与 inertia_calibrated.yaml 一致）
CALIBRATION_LINKS = ("link1", "link2", "link3", "link4", "link5", "link6")
ARM_NUM_JOINTS = 6
# 锁定夹爪关节，构建 6-DOF 臂模型
GRIPPER_JOINTS = ("joint_left", "joint_right")

# --------------------------------------------------------------------------- #
# 内嵌标定参数（连杆质量[kg] + 质心[m]，零位力矩偏置[N·m]）
# 源自 config/inertia_calibrated.yaml（已内联，无需外部文件）。
# 静态重力补偿仅依赖一阶矩 h = mass·com；如需更新请重新标定后替换此处。
# --------------------------------------------------------------------------- #
CALIBRATED_INERTIA = {
    "link1": {"mass": 0.4655, "com": (0.008176522803163536, 0.0012258421192189702, 0.03314196589672953)},
    "link2": {"mass": 1.972,  "com": (-0.5166599992840496, 0.07198069330805953, -0.027155193850031335)},
    "link3": {"mass": 1.062,  "com": (0.5664955065347711, -0.28030612137475797, -0.02710686877164231)},
    "link4": {"mass": 0.66,   "com": (0.22409003165272148, -0.0714393787048434, -0.031411192491840524)},
    "link5": {"mass": 0.1501, "com": (0.023331489573750076, 0.02146874732328794, -0.08211448245408905)},
    "link6": {"mass": 0.8304, "com": (-0.005366267871267582, -0.005638563740544155, 0.06327051501838667)},
}

JOINT_TORQUE_BIAS = (
    0.0,
    -0.9393265972173288,
    0.29461078326807044,
    -0.5456422186362749,
    0.002997027692182997,
    0.04110218256983898,
)


# --------------------------------------------------------------------------- #
# 路径工具
# --------------------------------------------------------------------------- #

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_config_dir() -> Path:
    return _repo_root() / "config"


def default_urdf_path() -> Path:
    return _repo_root() / "urdf" / "00-arm-rs_asm-v3" / "urdf" / "00-arm-rs_asm-v3.urdf"


# --------------------------------------------------------------------------- #
# 重力模型构建（与标定严格一致：缩减 6-DOF 模型 + 力矩回归量）
# --------------------------------------------------------------------------- #

def build_reduced_arm_model(urdf_path: Optional[str] = None) -> pin.Model:
    """从 URDF 加载并锁定夹爪关节，得到 6-DOF 臂模型（nv=6）。"""
    path = urdf_path or str(default_urdf_path())
    model = pin.buildModelFromUrdf(path)
    lock_ids = [model.getJointId(name) for name in GRIPPER_JOINTS]
    q_ref = pin.neutral(model)
    return pin.buildReducedModel(model, lock_ids, q_ref)


def _link_to_body_id(model: pin.Model, link_name: str) -> int:
    frame_id = model.getFrameId(link_name)
    if frame_id >= model.nframes:
        raise ValueError(f"未找到 frame/link: {link_name}")
    return int(model.frames[frame_id].parentJoint)


def _body_param_base(body_id: int) -> int:
    return (body_id - 1) * 10


def _phi_from_inertia(model: pin.Model) -> np.ndarray:
    """从模型惯量提取每连杆 10 参数向量 phi（默认 URDF 名义值）。"""
    n_param = (model.nbodies - 1) * 10
    phi = np.zeros(n_param)
    for body_id in range(1, model.nbodies):
        base = _body_param_base(body_id)
        I = model.inertias[body_id]
        phi[base] = I.mass
        h = I.mass * I.lever
        phi[base + 1 : base + 4] = h
        phi[base + 4] = I.inertia[0, 0]
        phi[base + 5] = I.inertia[0, 1]
        phi[base + 6] = I.inertia[1, 1]
        phi[base + 7] = I.inertia[0, 2]
        phi[base + 8] = I.inertia[1, 2]
        phi[base + 9] = I.inertia[2, 2]
    return phi


def build_gravity_regressor(model: pin.Model, data: pin.Data, q: np.ndarray) -> np.ndarray:
    """静态（v=0, a=0）关节力矩回归量 Y(q)，满足 tau = Y(q) @ phi。"""
    q = np.asarray(q, dtype=float).reshape(model.nq)
    v = np.zeros(model.nv)
    a = np.zeros(model.nv)
    pin.computeJointTorqueRegressor(model, data, q, v, a)
    return data.jointTorqueRegressor.copy()


def load_calibrated_phi(
    model: pin.Model,
    use_calibrated: bool,
) -> Tuple[np.ndarray, list, str]:
    """返回 (phi, joint_torque_bias, 模型描述)。

    use_calibrated=True 时使用内嵌的标定质量/质心覆盖模型一阶矩 h=m·com；
    否则直接使用 URDF 名义惯量。所有参数已内联，无需外部 YAML。
    """
    phi = _phi_from_inertia(model)
    bias_raw: list = []
    if not use_calibrated:
        return phi, bias_raw, "URDF 名义 [缩减6-DOF]"

    for lk in CALIBRATION_LINKS:
        if lk in CALIBRATED_INERTIA:
            base = _body_param_base(_link_to_body_id(model, lk))
            m = float(CALIBRATED_INERTIA[lk]["mass"])
            com = np.array(CALIBRATED_INERTIA[lk]["com"], dtype=float)
            phi[base] = m
            phi[base + 1 : base + 4] = m * com
    bias_raw = list(JOINT_TORQUE_BIAS)
    return phi, bias_raw, "标定 (内嵌参数) [缩减6-DOF, 与标定一致]"


# --------------------------------------------------------------------------- #
# 输出端扭矩逆变换
# --------------------------------------------------------------------------- #

def load_motor_torque_calib(arm_yaml: Path, calib_yaml: Path, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """返回逐臂关节 (gain[n], offset[n])，用于输出端逆变换。

    电机实际输出 = gain*指令 + offset*sign(指令)。
    """
    gain = np.ones(n)
    offset = np.zeros(n)
    if not (arm_yaml.is_file() and calib_yaml.is_file()):
        return gain, offset
    with open(arm_yaml, "r", encoding="utf-8") as f:
        models = [str(j.get("model", "")) for j in (yaml.safe_load(f) or {}).get("joints", [])]
    with open(calib_yaml, "r", encoding="utf-8") as f:
        cfg = (yaml.safe_load(f) or {}).get("models", {})
    for i in range(min(n, len(models))):
        mc = cfg.get(models[i], {})
        gain[i] = float(mc.get("gain", 1.0))
        offset[i] = float(mc.get("offset", 0.0))
    return gain, offset


def inverse_torque(tau_des: np.ndarray, gain: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """求指令使电机实际输出 = tau_des：cmd = sign(τ)·max(0,|τ|−b)/a。

    |τ|<=b 时进入死区(指令为0)，宁可轻微欠补偿(臂略下沉)也不过补偿(上抬)。
    """
    s = np.sign(tau_des)
    mag = np.maximum(0.0, np.abs(tau_des) - offset) / gain
    return s * mag


# --------------------------------------------------------------------------- #
# 自适应阻尼 kd
# --------------------------------------------------------------------------- #

def compute_adaptive_kd(
    vel: np.ndarray,
    prev_kd: np.ndarray,
    kd_min: np.ndarray,
    kd_max: np.ndarray,
    v_ref: float,
    alpha: float,
) -> np.ndarray:
    """根据关节速度计算自适应 MIT 阻尼 kd（低速粘滞、高速顺滑，EMA 平滑）。"""
    kd_raw = kd_min + (kd_max - kd_min) / (1.0 + (np.abs(vel) / v_ref) ** 2)
    return alpha * kd_raw + (1.0 - alpha) * prev_kd


def _sigint_handler(signum, frame):
    global _running
    print("\n[zero_torque] 收到 Ctrl+C，准备停止...")
    _running = False


def main() -> None:
    parser = argparse.ArgumentParser(description="零力矩 / 自由拖动模式（单文件，仅执行）")
    parser.add_argument("--model", choices=["cal", "urdf"], default="cal",
                        help="重力模型：cal=标定惯量(默认)，urdf=名义")
    parser.add_argument("--kd", type=str, default="0.6",
                        help="关节阻尼：单值(全局)或逗号分隔逐关节"
                             "(如 0.6,1,0.6,0.6,0.6,0.6)；越大越粘滞")
    parser.add_argument("--adaptive-kd", action="store_true",
                        help="开启自适应阻尼：低速接近 kd_max，高速接近 kd_min")
    parser.add_argument("--kd-min", type=str, default="0.001",
                        help="自适应 kd 下限：单值或逐关节，默认 0.001")
    parser.add_argument("--kd-max", type=str, default="0.15",
                        help="自适应 kd 上限：单值或逐关节，默认 0.15")
    parser.add_argument("--per-joint-kd-min", type=str, default="",
                        help="逐关节覆盖 kd_min，留空则使用 --kd-min")
    parser.add_argument("--per-joint-kd-max", type=str, default="",
                        help="逐关节覆盖 kd_max，留空则使用 --kd-max")
    parser.add_argument("--kd-v-ref", type=float, default=1.0,
                        help="自适应 kd 速度参考值(rad/s)，默认 1.0")
    parser.add_argument("--kd-smoothing-alpha", type=float, default=0.15,
                        help="自适应 kd 一阶平滑系数[0,1]，默认 0.15")
    parser.add_argument("--ramp", type=float, default=1.5,
                        help="重力前馈缓升时间（秒，默认 1.5）")
    parser.add_argument("--scale", type=str, default="1.0",
                        help="重力前馈缩放：单值(全局)或逗号分隔逐关节"
                             "(如 1,1,0.93,0.93,1,1)；<1 欠补偿更安全")
    parser.add_argument("--exclude-joints", type=str, default="1",
                        help="不参与零力矩的关节(1-based,逗号分隔)，前馈置0。"
                             "默认 1（竖直轴 joint1 不承重）。设空字符串关闭")
    parser.add_argument("--no-output-calib", action="store_true",
                        help="关闭输出端扭矩逆变换(默认开启，抵消电机增益+偏置)")
    parser.add_argument("--no-bias", action="store_true",
                        help="关闭零位力矩偏置补偿(默认读取标定文件 joint_torque_bias)")
    args = parser.parse_args()

    exclude_joints = [int(s) - 1 for s in args.exclude_joints.split(",") if s.strip()]
    scale_list = [float(s) for s in args.scale.split(",") if s.strip()] or [1.0]
    kd_list = [float(s) for s in args.kd.split(",") if s.strip()] or [0.6]
    kd_min_list = [float(s) for s in args.kd_min.split(",") if s.strip()] or [0.001]
    kd_max_list = [float(s) for s in args.kd_max.split(",") if s.strip()] or [0.15]
    per_joint_kd_min_list = [float(s) for s in args.per_joint_kd_min.split(",") if s.strip()]
    per_joint_kd_max_list = [float(s) for s in args.per_joint_kd_max.split(",") if s.strip()]
    if args.kd_v_ref <= 1e-6:
        print("[错误] --kd-v-ref 必须 > 0")
        sys.exit(1)
    if not 0.0 <= args.kd_smoothing_alpha <= 1.0:
        print("[错误] --kd-smoothing-alpha 必须在 [0, 1] 内")
        sys.exit(1)

    signal.signal(signal.SIGINT, _sigint_handler)

    # 重力模型：缩减 6-DOF 模型 + 力矩回归量（与标定一致，避免重复计夹爪质量）
    red_model = build_reduced_arm_model()
    red_data = red_model.createData()
    phi_cal, bias_raw, model_name = load_calibrated_phi(
        red_model, use_calibrated=(args.model == "cal")
    )
    if args.no_bias:
        bias_raw = []

    excl_txt = ("，".join(f"joint{j+1}" for j in exclude_joints) or "无")
    print("=" * 60)
    print("  零力矩 / 自由拖动模式（单文件，仅执行）")
    print(f"  重力模型: {model_name}")
    kd_mode = "自适应" if args.adaptive_kd else "固定"
    print(f"  kp=0  kd={args.kd} ({kd_mode})  缓升={args.ramp}s  缩放={args.scale}")
    print(f"  不参与零力矩(前馈置0): {excl_txt}")
    print("  行为: 机械臂失重，可用手拖动；松手停在原地")
    print("  Ctrl+C 退出（自动失能）")
    print("=" * 60)

    arm = RobotArm()
    arm.connect()
    print("[连接] OK")

    # 输出端扭矩逆变换系数（抵消电机 实际=a·指令+b·sign）
    if args.no_output_calib:
        out_gain = np.ones(arm.num_joints)
        out_offset = np.zeros(arm.num_joints)
        print("[输出修正] 关闭")
    else:
        out_gain, out_offset = load_motor_torque_calib(
            default_config_dir() / "arm.yaml",
            default_config_dir() / "motor_torque_calib.yaml",
            arm.num_joints,
        )
        print("[输出修正] 逆变换 cmd=sign(g)·max(0,|g|-b)/a")
        for i in range(arm.num_joints):
            print(f"  joint{i+1}: a={out_gain[i]:.4f} b={out_offset[i]:+.4f}")

    arm.enable()
    print("[使能] OK")
    n_j = arm.num_joints

    def _broadcast(lst, default):
        if len(lst) == 1:
            return np.full(n_j, lst[0])
        a = np.full(n_j, default)
        k = min(n_j, len(lst))
        a[:k] = lst[:k]
        return a

    kd_arr = _broadcast(kd_list, 0.6)
    kd_min_arr = _broadcast(per_joint_kd_min_list or kd_min_list, 0.001)
    kd_max_arr = _broadcast(per_joint_kd_max_list or kd_max_list, 0.15)
    if np.any(kd_min_arr > kd_max_arr):
        print("[错误] kd_min 不能大于 kd_max")
        sys.exit(1)
    scale_arr = _broadcast(scale_list, 1.0)
    bias_arr = _broadcast([float(x) for x in bias_raw], 0.0) if bias_raw else np.zeros(n_j)
    if args.adaptive_kd:
        kd_arr = np.clip(kd_arr, kd_min_arr, kd_max_arr)
        print(f"[自适应 kd]    min={np.round(kd_min_arr, 4).tolist()}"
              f" max={np.round(kd_max_arr, 4).tolist()}"
              f" v_ref={args.kd_v_ref:g} alpha={args.kd_smoothing_alpha:g}")
        print(f"[初始 kd]      {np.round(kd_arr, 4).tolist()}")
    else:
        print(f"[逐关节 kd]    {np.round(kd_arr, 3).tolist()}")
    print(f"[逐关节 scale] {np.round(scale_arr, 3).tolist()}")
    print(f"[零位偏置]     {np.round(bias_arr, 3).tolist()} N·m"
          + ("" if bias_raw else "  (无/关闭)"))

    arm.mode_mit(kp=np.zeros(n_j), kd=kd_arr)

    state = {"t0": time.time(), "n": 0, "kd": kd_arr.copy()}

    def controller(arm_ref: RobotArm, dt: float) -> None:
        q, qd, _ = arm_ref.get_state()
        # 与标定一致：缩减模型回归量 Y(q) @ phi_cal
        Y = build_gravity_regressor(red_model, red_data, q)
        tau_g = (Y @ phi_cal)[: arm_ref.num_joints]
        if args.adaptive_kd:
            state["kd"] = compute_adaptive_kd(
                qd,
                state["kd"],
                kd_min_arr,
                kd_max_arr,
                args.kd_v_ref,
                args.kd_smoothing_alpha,
            )
        current_kd = state["kd"] if args.adaptive_kd else kd_arr

        elapsed = time.time() - state["t0"]
        ramp = min(1.0, elapsed / args.ramp) if args.ramp > 1e-3 else 1.0
        # 期望电机实际输出 = g(q)+零位偏置（复现静止保持力矩），缩放+缓升后做逆变换
        tau_des = (tau_g + bias_arr) * scale_arr * ramp
        tau_cmd = inverse_torque(tau_des, out_gain, out_offset)
        for j in exclude_joints:
            if 0 <= j < tau_cmd.shape[0]:
                tau_cmd[j] = 0.0

        arm_ref.mit(
            pos=q,
            vel=np.zeros(arm_ref.num_joints),
            kp=np.zeros(arm_ref.num_joints),
            kd=current_kd,
            tau=tau_cmd,
            request_feedback=True,
        )

        state["n"] += 1
        if state["n"] % 50 == 0:
            tag = "缓升" if ramp < 1.0 else "运行"
            print(f"[{state['n']:5d}|{tag}] g={' '.join(f'{t:+.2f}' for t in tau_des)}"
                  f" | cmd={' '.join(f'{t:+.2f}' for t in tau_cmd)} N·m"
                  f" | kd={' '.join(f'{k:.3f}' for k in current_kd)}")

    arm.start_control_loop(controller, rate=arm._rate)
    print(f"[控制循环] @ {arm._rate} Hz\n" + "-" * 60)

    try:
        while _running:
            time.sleep(0.02)
    finally:
        print("\n[停止] 关闭控制循环并失能...")
        arm.disconnect()
        print("[完成] 已安全断开")


if __name__ == "__main__":
    main()
