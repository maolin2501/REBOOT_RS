"""reBotArm 手柄遥操作控制器（增量式笛卡尔空间）。

参考 EL-A3 ``XboxArmController`` 的控制流程，适配到 reBotArm 的
``RobotArm`` / ``kinematics`` / ``dynamics`` 接口：

    手柄 → 末端 6 维速度 → 死区 + EMA 平滑 → 增量积分到目标位姿
        → 阻尼最小二乘 IK（带跳变保护与自动重同步）
        → 二阶临界阻尼滤波 → MIT 关节指令下发（位置 + 速度前馈 + 重力前馈）

与 EL-A3 的主要差异（接口适配）：

================  =======================  ========================================
环节              EL-A3 (ELA3Interface)    reBotArm (RobotArm)
================  =======================  ========================================
下发              ``JointCtrl``            ``mit(pos, vel, kp, kd, tau)``
回零/Home         ``MoveJ`` 阻塞           控制环内消费 min-jerk 关节轨迹缓冲
零力矩            ``ZeroTorqueMode()``     ``kp=0`` + 重力前馈力矩 ``tau=g(q)``
急停              ``EmergencyStop()``      软冻结（锁定当前位置保持）
反馈              ``GetArmJointMsgs()``    ``get_state()``
正/逆运动学       ``kin.fk/ik_step``       6-DOF 缩减模型 FK / 增量 DLS-IK + 跳变保护
重力补偿          SDK 内置                 ``compute_generalized_gravity`` + 输出端逆变换
================  =======================  ========================================

末端跟踪点（TCP）取 joint6 坐标系前向（局部 +Z）150mm 处；运动学使用锁定夹爪
(joint_left/joint_right) 的 6-DOF 缩减模型，与机械臂的 6 个臂关节一一对应。

整个控制在 ``RobotArm.start_control_loop`` 的单一后台线程内按 ``update_rate``
（默认 100 Hz）运行，所有 CAN 访问串行化，无多线程竞争。
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pinocchio as pin
import yaml

from ..actuator import RobotArm, Gripper
from ..kinematics import (
    load_robot_model,
    pos_rot_to_se3,
)
from ..dynamics import load_dynamics_model, compute_generalized_gravity

from .joystick import LinuxJoystick
from .controller_profiles import ControllerProfile

logger = logging.getLogger("reBotArm.teleop")

# 速度档位：(显示名, 缩放系数)
SPEED_LEVELS = [
    ("极慢", 0.10),
    ("慢",   0.25),
    ("中",   0.50),
    ("快",   0.75),
    ("最大", 1.00),
]

# Home 位姿取自 config/calibration_poses.yaml 的 home_q。
# 注意：joint2/joint3 的 URDF 限位为 [0, π]，home 必须取正角（不可用负值，否则被限位钳裁）。
HOME_POSITIONS = [0.0, 1.0, 1.2, 0.0, 0.0, 0.0]
ZERO_POSITIONS = [0.0] * 6


def _inverse_output_torque(
    tau_des: np.ndarray, gain: np.ndarray, offset: np.ndarray,
) -> np.ndarray:
    """电机输出端逆变换：求指令使实际输出 ≈ tau_des。

    电机输出模型 ``tau_actual = gain·cmd + offset·sign(cmd)``，其逆为
    ``cmd = sign(τ)·max(0, |τ|−offset) / gain``。``|τ| ≤ offset`` 落入死区
    （指令为 0），宁可轻微欠补偿也不过补偿。与 ``ArmEndPos`` 保持一致。
    """
    s = np.sign(tau_des)
    mag = np.maximum(0.0, np.abs(tau_des) - offset) / gain
    return s * mag


class XboxTeleopController:
    """基于 RobotArm 的手柄机械臂遥操作控制器（末端坐标模式）。"""

    def __init__(
        self,
        arm: RobotArm,
        joystick: LinuxJoystick,
        profile: ControllerProfile,
        gripper: Optional[Gripper] = None,
        update_rate: float = 100.0,
        max_linear_velocity: float = 0.15,
        max_angular_velocity: float = 1.5,
        deadzone: float = 0.15,
        input_smoothing: float = 0.35,
        filter_omega: float = 14.0,
        max_ik_jump: float = 0.5,
        tcp_offset: float = 0.15,
        ik_damping: float = 5e-3,
        gravity_ff: bool = True,
        gravity_ff_ratio: float = 0.95,
        output_torque_calib: bool = True,
        zero_torque_kd: float = 0.5,
        home_velocity: float = 0.5,
        startup_ramp: float = 1.0,
    ):
        self._arm = arm
        self._joy = joystick
        self._gripper = gripper
        self._profile = profile
        self._rate = update_rate
        self._dt = 1.0 / update_rate
        self._n = arm.num_joints

        self._max_lin_vel = max_linear_velocity
        self._max_ang_vel = max_angular_velocity
        self._dz_threshold = deadzone
        self._input_alpha = input_smoothing
        self._filter_omega = filter_omega
        self._max_ik_jump = max_ik_jump
        self._zt_kd = float(zero_torque_kd)
        self._home_vel = float(home_velocity)
        self._startup_ramp = float(startup_ramp)

        # ── 运动学模型：锁定夹爪的 6-DOF 缩减模型 + TCP 帧（joint6 前向 150mm）─
        self._tcp_offset = float(tcp_offset)
        self._model, self._end_frame_id = self._build_arm_model_with_tcp(self._tcp_offset)
        self._data = self._model.createData()
        # 增量 IK 参数（固定阻尼 DLS + 步长截断，奇异稳定，适合实时遥操作）。
        # 注：不使用 kinematics.solve_ik —— 其回退线搜索为离线一次性求解设计，
        # 在实时增量场景下会拒绝合理步长导致关节不更新。
        self._ik_damping = 5e-3      # DLS 阻尼 λ
        self._ik_iters = 8           # 每帧最大迭代数（种子接近时 1-3 步即收敛）
        self._ik_tol = 1e-4          # 收敛阈值 ||err6||
        self._ik_max_substep = 0.2   # 单次迭代关节步长上限 (rad)，抑制奇异跳变
        _lo = np.array(self._model.lowerPositionLimit, dtype=np.float64)
        _hi = np.array(self._model.upperPositionLimit, dtype=np.float64)
        _lo[~np.isfinite(_lo)] = -np.inf
        _hi[~np.isfinite(_hi)] = np.inf
        self._jlim_lo, self._jlim_hi = _lo, _hi

        # ── MIT 增益（来自 arm.yaml 逐关节 kp/kd）──────────────────────────
        self._mit_kp = np.array([j.kp for j in arm._joints], dtype=np.float64)
        self._mit_kd = np.array([j.kd for j in arm._joints], dtype=np.float64)

        # ── 重力前馈（标定惯量 + 输出端力矩逆变换，与 ArmEndPos 一致）──────
        self._gravity_ff = bool(gravity_ff)
        self._gravity_ff_ratio = float(gravity_ff_ratio)
        self._output_torque_calib = bool(output_torque_calib)
        self._dyn_model = None
        self._dyn_data = None
        self._out_gain = np.ones(self._n)
        self._out_offset = np.zeros(self._n)
        if self._gravity_ff:
            self._setup_gravity_model()

        # 速度档位
        self._speed_idx = 2
        self._speed_factor = SPEED_LEVELS[self._speed_idx][1]

        # 模式状态
        self._running = False
        self._zero_torque = False
        self._is_moving = False
        self._exit_requested = False
        self._estop = False
        self._t_start = 0.0

        # 末端位姿跟踪（np6: [x, y, z, roll, pitch, yaw]）
        self._target_pose: Optional[np.ndarray] = None
        self._prev_pose: Optional[np.ndarray] = None

        # IK / 滤波状态
        self._ik_seed: np.ndarray = np.zeros(self._n)
        self._ik_filter_pos: np.ndarray = np.zeros(self._n)
        self._ik_filter_vel: np.ndarray = np.zeros(self._n)
        self._ik_raw: Optional[np.ndarray] = None
        self._consecutive_rejects = 0
        self._consecutive_ik_fails = 0
        self._seed_just_init = False
        self._resync_cooldown = 0

        # 输入 EMA 状态（末端 6 维速度）
        self._sv = np.zeros(6)  # [vx, vy, vz, wroll, wpitch, wyaw]

        # 急停冻结位置
        self._q_freeze = np.zeros(self._n)

        # 回 Home / 零位 的 min-jerk 轨迹缓冲（控制环逐点消费）
        self._move_buffer: List[tuple] = []
        self._move_idx = 0
        self._move_name = ""

        # 夹爪
        self._gripper_angle = 0.0
        self._gripper_step = 0.2
        self._gripper_min = -1.5708
        self._gripper_max = 1.5708

        # 按钮边沿检测
        self._prev_btn = [0] * LinuxJoystick.MAX_BUTTONS
        self._prev_dpad_up = 0
        self._prev_dpad_down = 0

        # 诊断
        self._diag_tick = 0

    @property
    def exit_requested(self) -> bool:
        return self._exit_requested

    # ------------------------------------------------------------------
    # 重力前馈模型
    # ------------------------------------------------------------------

    def _setup_gravity_model(self) -> None:
        cfg_dir = Path(__file__).resolve().parents[2] / "config"
        cal_yaml = cfg_dir / "inertia_calibrated.yaml"
        if cal_yaml.is_file():
            self._dyn_model = load_dynamics_model(calibration_yaml=str(cal_yaml))
            logger.info("重力前馈: 标定惯量 %s ×%.2f", cal_yaml.name, self._gravity_ff_ratio)
        else:
            self._dyn_model = load_dynamics_model()
            logger.info("重力前馈: URDF 名义惯量（未找到标定文件）×%.2f", self._gravity_ff_ratio)
        self._dyn_data = self._dyn_model.createData()

        if self._output_torque_calib:
            self._out_gain, self._out_offset = self._load_output_torque_calib(
                cfg_dir / "motor_torque_calib.yaml")

    def _load_output_torque_calib(self, calib_yaml: Path) -> tuple:
        gain = np.ones(self._n)
        offset = np.zeros(self._n)
        if not calib_yaml.is_file():
            logger.warning("未找到 %s，输出端逆变换按恒等处理", calib_yaml.name)
            return gain, offset
        with open(calib_yaml, "r", encoding="utf-8") as f:
            models = (yaml.safe_load(f) or {}).get("models", {})
        for i, jc in enumerate(self._arm._joints):
            mc = models.get(jc.model, {})
            gain[i] = float(mc.get("gain", 1.0))
            offset[i] = float(mc.get("offset", 0.0))
        return gain, offset

    def _gravity_tau(self, q: np.ndarray) -> np.ndarray:
        """重力补偿前馈力矩 = 输出端逆变换(ratio · g(q))。"""
        if not self._gravity_ff or self._dyn_model is None:
            return np.zeros(self._n)
        q_full = np.asarray(q, dtype=np.float64).reshape(-1)
        if q_full.shape[0] != self._dyn_model.nq:
            qf = np.zeros(self._dyn_model.nq)
            k = min(self._dyn_model.nq, q_full.shape[0])
            qf[:k] = q_full[:k]
            q_full = qf
        g = compute_generalized_gravity(self._dyn_model, q_full, self._dyn_data)
        tau = self._gravity_ff_ratio * g
        if self._output_torque_calib:
            n = min(self._n, tau.shape[0])
            tau[:n] = _inverse_output_torque(
                tau[:n], self._out_gain[:n], self._out_offset[:n])
        return tau[:self._n]

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self) -> None:
        """以当前实际位置为基准建立目标位姿，并启动 RobotArm 控制循环。"""
        self._running = True
        self._initialize_baseline()
        self._print_banner()
        self._t_start = time.monotonic()
        self._arm.start_control_loop(self._tick, rate=self._rate)

    def stop(self) -> None:
        self._running = False
        self._arm.stop_control_loop()

    def _initialize_baseline(self) -> None:
        """读取当前关节角，用 FK 设初始目标位姿（避免上电猛冲）。"""
        q0 = self._arm.get_positions(request=True)
        q0 = np.asarray(q0, dtype=np.float64).reshape(-1)[: self._n]
        self._ik_seed = q0.copy()
        self._ik_filter_pos = q0.copy()
        self._ik_filter_vel = np.zeros(self._n)
        self._ik_raw = q0.copy()
        self._q_freeze = q0.copy()
        self._seed_just_init = True
        self._consecutive_rejects = 0
        self._consecutive_ik_fails = 0
        self._target_pose = self._fk_pose(q0)
        self._prev_pose = None
        p = self._target_pose
        logger.info("初始化完成, 末端位姿: (%.3f, %.3f, %.3f) m  (%.2f, %.2f, %.2f) rad",
                    p[0], p[1], p[2], p[3], p[4], p[5])

        # 夹爪初始角度对齐到当前实际位置，避免首次 D-pad 操作突跳
        if self._gripper is not None:
            try:
                self._gripper_angle = float(self._gripper.get_position(request=True))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 运动学模型与辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _build_arm_model_with_tcp(tcp_offset: float):
        """构建末端遥操作用的运动学模型。

        完整 URDF 含 8 个自由度（6 臂关节 + joint_left/joint_right 两个夹爪手指），
        遥操作只控制臂的 6 关节，故锁定夹爪关节得到 6-DOF 缩减模型；并在 joint6
        坐标系前向（局部 +Z）``tcp_offset`` 米处添加 TCP 操作帧作为跟踪控制点。

        返回 ``(model, tcp_frame_id)``。
        """
        full = load_robot_model()
        lock_ids = [full.getJointId("joint_left"), full.getJointId("joint_right")]
        model = pin.buildReducedModel(full, lock_ids, pin.neutral(full))
        j6_id = model.getJointId("joint6")
        parent_frame = model.getFrameId("link6")
        placement = pin.SE3(np.eye(3), np.array([0.0, 0.0, float(tcp_offset)]))
        tcp = pin.Frame("tcp", j6_id, parent_frame, placement, pin.FrameType.OP_FRAME)
        frame_id = model.addFrame(tcp)
        return model, frame_id

    def _fk_pose(self, q) -> np.ndarray:
        """关节角 → TCP 末端位姿 [x, y, z, roll, pitch, yaw]。"""
        q = np.asarray(q, dtype=np.float64).reshape(-1)[: self._model.nq]
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)
        T = self._data.oMf[self._end_frame_id]
        rpy = pin.rpy.matrixToRpy(T.rotation)
        return np.concatenate([T.translation.copy(), rpy])

    def _ik_step(self, target_pose: np.ndarray, seed: np.ndarray) -> tuple:
        """增量阻尼最小二乘 IK（固定阻尼 + 步长截断）。

        以上一帧解为种子，对 TCP 目标位姿做少量 DLS 迭代：误差 twist 取 LOCAL 系
        ``log6(T_cur⁻¹·T_target)``，配 LOCAL 系帧雅可比；固定阻尼保证奇异点附近数值
        稳定，单次迭代步长截断抑制跳变。返回 ``(q_sol, err, converged)``。
        """
        target = pos_rot_to_se3(
            target_pose[:3], roll=float(target_pose[3]),
            pitch=float(target_pose[4]), yaw=float(target_pose[5]))
        model, data, fid = self._model, self._data, self._end_frame_id
        lam = self._ik_damping
        eye6 = np.eye(6)
        q = np.asarray(seed, dtype=np.float64).copy()
        err = float("inf")
        for _ in range(self._ik_iters):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            e6 = pin.log6(data.oMf[fid].inverse() * target).vector
            err = float(np.linalg.norm(e6))
            if err < self._ik_tol:
                break
            pin.computeJointJacobians(model, data, q)
            J = pin.getFrameJacobian(model, data, fid, pin.LOCAL)
            dq = J.T @ np.linalg.solve(J @ J.T + lam * eye6, e6)
            mx = float(np.max(np.abs(dq)))
            if mx > self._ik_max_substep:
                dq *= self._ik_max_substep / mx
            q = pin.integrate(model, q, dq)
            q = np.minimum(np.maximum(q, self._jlim_lo), self._jlim_hi)
        return q, err, err < self._ik_tol

    # ------------------------------------------------------------------
    # 输入辅助
    # ------------------------------------------------------------------

    def _apply_dz(self, val: float) -> float:
        if abs(val) < self._dz_threshold:
            return 0.0
        sign = 1.0 if val > 0 else -1.0
        return sign * (abs(val) - self._dz_threshold) / (1.0 - self._dz_threshold)

    def _apply_trigger(self, raw: float) -> float:
        norm = max(0.0, min(raw, 1.0))
        dz = self._dz_threshold * 1.5
        if norm < dz:
            return 0.0
        return (norm - dz) / (1.0 - dz)

    def _axis_value(self, binding) -> float:
        return binding.read(self._joy.axes)

    def _trigger_value(self, binding) -> float:
        return binding.read(self._joy.axes, self._joy.buttons)

    def _button_state(self, idx: Optional[int]) -> int:
        if idx is None or idx >= len(self._joy.buttons):
            return 0
        return self._joy.buttons[idx]

    def _btn_edge(self, idx: Optional[int]) -> bool:
        if idx is None or idx >= len(self._joy.buttons):
            return False
        return self._joy.buttons[idx] == 1 and self._prev_btn[idx] == 0

    # ------------------------------------------------------------------
    # 主控制循环 tick（由 RobotArm.start_control_loop 以 update_rate 调用）
    # ------------------------------------------------------------------

    def _tick(self, _arm: RobotArm, _dt: float) -> None:
        try:
            self._tick_impl()
        except Exception as e:  # 单帧异常不得中断控制循环
            logger.error("控制循环异常: %s", e)

    def _tick_impl(self) -> None:
        if not self._joy.connected:
            # 手柄断开：冻结保持，避免失控
            self._hold(self._ik_filter_pos)
            return

        buttons = self._profile.buttons
        sticks = self._profile.sticks

        # ---- 按钮事件（边沿触发，任何模式下都响应）----
        if self._btn_edge(buttons.south):
            self._speed_idx = (self._speed_idx + 1) % len(SPEED_LEVELS)
            self._speed_factor = SPEED_LEVELS[self._speed_idx][1]
            self._log_speed()

        if self._btn_edge(buttons.east):
            self._start_move(HOME_POSITIONS, "Home")

        if self._btn_edge(buttons.west):
            self._start_move(ZERO_POSITIONS, "零位")

        if self._btn_edge(buttons.north):
            self._toggle_zero_torque()

        if self._btn_edge(buttons.back):
            self._toggle_estop()

        if self._btn_edge(buttons.start):
            logger.info("收到退出请求")
            self._exit_requested = True

        # D-pad 夹爪
        dpad_y = self._axis_value(sticks.dpad_y)
        dpad_up = 1 if dpad_y < -0.5 else 0
        dpad_down = 1 if dpad_y > 0.5 else 0
        if dpad_up and not self._prev_dpad_up:
            self._gripper_angle = min(self._gripper_angle + self._gripper_step, self._gripper_max)
            self._gripper_cmd(self._gripper_angle)
        if dpad_down and not self._prev_dpad_down:
            self._gripper_angle = max(self._gripper_angle - self._gripper_step, self._gripper_min)
            self._gripper_cmd(self._gripper_angle)
        self._prev_dpad_up = dpad_up
        self._prev_dpad_down = dpad_down

        self._prev_btn = list(self._joy.buttons)

        # ---- 模式分支（每帧都必须下发，否则电机失指令）----
        if self._estop:
            self._hold(self._q_freeze)
            self._periodic_status()
            return

        if self._zero_torque:
            self._zero_torque_hold()
            self._periodic_status()
            return

        if self._is_moving:
            self._consume_move_buffer()
            self._periodic_status()
            return

        # ---- 正常遥操作：末端速度映射 ----
        max_lin = self._max_lin_vel * self._speed_factor
        max_ang = self._max_ang_vel * self._speed_factor

        raw = np.array([
            -self._apply_dz(self._axis_value(sticks.ly)) * max_lin,                 # vx
            -self._apply_dz(self._axis_value(sticks.lx)) * max_lin,                 # vy
            (self._apply_trigger(self._trigger_value(sticks.rt))
             - self._apply_trigger(self._trigger_value(sticks.lt))) * max_lin,      # vz
            self._apply_dz(self._axis_value(sticks.ry)) * max_ang,                  # wroll
            (self._button_state(buttons.rb)
             - self._button_state(buttons.lb)) * max_ang,                           # wpitch
            self._apply_dz(self._axis_value(sticks.rx)) * max_ang,                  # wyaw
        ])

        # EMA 平滑 + 对称释放衰减
        total = float(np.sum(np.abs(raw)))
        if total < 1e-6:
            decay = min(self._input_alpha * 3.0, 1.0)
            self._sv = (1.0 - decay) * self._sv
        else:
            a = self._input_alpha
            self._sv = a * raw + (1.0 - a) * self._sv

        # 重同步冷却：让系统稳定数帧
        if self._resync_cooldown > 0:
            self._resync_cooldown -= 1
            self._send_filtered()
            self._periodic_status()
            return

        # ---- 位姿积分 + 增量 IK ----
        sv = self._sv
        has_input = float(np.sum(np.abs(sv))) > 1e-7

        if has_input:
            self._prev_pose = self._target_pose.copy()
            dt = self._dt
            self._target_pose[:3] += sv[:3] * dt
            self._target_pose[3:] += sv[3:] * dt

            try:
                q_sol, ik_err, success = self._ik_step(self._target_pose, self._ik_seed)
                if self._accept_ik(q_sol):
                    self._ik_raw = q_sol.copy()
                    self._ik_seed = q_sol.copy()
                    if success:
                        self._consecutive_ik_fails = 0
                    else:
                        self._consecutive_ik_fails += 1

                    # 残差大：把目标位姿向实际可达 FK 位姿 blend 拉回
                    if ik_err > 0.01:
                        fk_pose = self._fk_pose(q_sol)
                        blend = min((ik_err - 0.01) * 20.0, 0.7)
                        self._target_pose += blend * (fk_pose - self._target_pose)

                    if self._consecutive_ik_fails >= 10:
                        logger.warning("IK 残差持续偏大 %d 帧 (err=%.4f)，目标可能接近工作空间边界",
                                       self._consecutive_ik_fails, ik_err)
                    if self._consecutive_ik_fails >= 50:
                        logger.warning("IK 连续不收敛 50+ 帧，自动重新同步...")
                        self._resync_ik()
                else:
                    self._target_pose = self._prev_pose
            except Exception as e:
                logger.error("IK 异常: %s", e)
                self._target_pose = self._prev_pose
        else:
            self._consecutive_ik_fails = 0

        # ---- 二阶临界阻尼滤波 → MIT 下发 ----
        self._send_filtered()
        self._periodic_status()

    # ------------------------------------------------------------------
    # IK 跳变保护与重同步
    # ------------------------------------------------------------------

    def _accept_ik(self, q_new: np.ndarray) -> bool:
        ref = self._ik_seed
        if ref is None:
            return True

        max_diff = float(np.max(np.abs(q_new - ref)))
        if max_diff <= self._max_ik_jump:
            if self._consecutive_rejects > 0:
                self._consecutive_rejects = 0
            self._seed_just_init = False
            return True

        # 种子刚初始化的第一帧豁免大跳变检查
        if self._seed_just_init:
            self._seed_just_init = False
            return True

        self._consecutive_rejects += 1
        if self._consecutive_rejects >= 5:
            logger.warning("疑似奇异区: IK 跳变=%.3frad, 已保护 %d 帧",
                           max_diff, self._consecutive_rejects)
        if self._consecutive_rejects >= 50:
            logger.warning("IK 连续拒绝 50+ 帧，自动重新同步...")
            self._resync_ik()
        return False

    def _read_averaged_feedback(self, n_samples: int = 5, interval: float = 0.004) -> np.ndarray:
        samples = []
        for _ in range(n_samples):
            q = self._arm.get_positions(request=True)
            samples.append(np.asarray(q, dtype=np.float64).reshape(-1)[: self._n])
            time.sleep(interval)
        return np.mean(samples, axis=0)

    def _resync_ik(self) -> None:
        """用多帧平均的真实反馈重置种子 / 滤波 / 目标位姿。"""
        q_avg = self._read_averaged_feedback()
        self._ik_seed = q_avg.copy()
        self._ik_filter_pos = q_avg.copy()
        self._ik_raw = q_avg.copy()
        self._ik_filter_vel *= 0.2
        self._seed_just_init = True
        self._consecutive_rejects = 0
        self._consecutive_ik_fails = 0
        self._resync_cooldown = 5
        self._target_pose = self._fk_pose(q_avg)
        self._prev_pose = None

    # ------------------------------------------------------------------
    # 二阶滤波 + MIT 下发
    # ------------------------------------------------------------------

    def _startup_gain_scale(self) -> float:
        """启动缓升系数：使能后位置刚度从 0 线性升到满量，避免上电冲击。"""
        if self._startup_ramp <= 1e-3:
            return 1.0
        elapsed = time.monotonic() - self._t_start
        return min(1.0, elapsed / self._startup_ramp)

    def _send_filtered(self) -> None:
        """二阶临界阻尼滤波（精确矩阵指数）→ MIT 下发位置 + 速度前馈 + 重力前馈。"""
        if self._ik_raw is None:
            return

        omega = self._filter_omega
        dt = self._dt
        a = omega * dt
        ea = math.exp(-a)
        for i in range(self._n):
            err = self._ik_raw[i] - self._ik_filter_pos[i]
            vel = self._ik_filter_vel[i]
            err_new = ea * ((1.0 + a) * err - dt * vel)
            vel_new = ea * (omega * omega * dt * err + (1.0 - a) * vel)
            self._ik_filter_pos[i] = self._ik_raw[i] - err_new
            self._ik_filter_vel[i] = vel_new

        scale = self._startup_gain_scale()
        tau = self._gravity_tau(self._ik_filter_pos)
        self._arm.mit(
            self._ik_filter_pos,
            vel=self._ik_filter_vel,
            kp=self._mit_kp * scale,
            kd=self._mit_kd,
            tau=tau,
        )

    def _hold(self, q: np.ndarray) -> None:
        """锁定保持在 q（急停 / 手柄断开），刚性位置 + 重力前馈。"""
        q = np.asarray(q, dtype=np.float64).reshape(-1)[: self._n]
        tau = self._gravity_tau(q)
        self._arm.mit(q, vel=np.zeros(self._n), kp=self._mit_kp, kd=self._mit_kd, tau=tau)

    def _zero_torque_hold(self) -> None:
        """零力矩 / 自由拖动：kp=0，仅重力前馈 + 少量阻尼，可手动拖动。"""
        q = self._arm.get_positions(request=False)
        q = np.asarray(q, dtype=np.float64).reshape(-1)[: self._n]
        tau = self._gravity_tau(q)
        self._arm.mit(
            q,
            vel=np.zeros(self._n),
            kp=np.zeros(self._n),
            kd=np.full(self._n, self._zt_kd),
            tau=tau,
        )

    # ------------------------------------------------------------------
    # 回 Home / 零位（min-jerk 关节轨迹，控制环逐点消费）
    # ------------------------------------------------------------------

    def _start_move(self, target_q: List[float], name: str) -> None:
        if self._is_moving:
            logger.warning("正在执行其他动作，请稍后再试")
            return
        # 解除零力矩 / 急停后再运动
        if self._zero_torque:
            self._zero_torque = False
        if self._estop:
            self._estop = False
        q_from = self._arm.get_positions(request=True)
        q_from = np.asarray(q_from, dtype=np.float64).reshape(-1)[: self._n]
        q_to = np.asarray(target_q, dtype=np.float64).reshape(-1)[: self._n]
        self._move_buffer = self._gen_min_jerk(q_from, q_to, self._home_vel)
        self._move_idx = 0
        self._move_name = name
        self._is_moving = True
        logger.info("正在移动到 %s ...", name)

    def _gen_min_jerk(self, q_from: np.ndarray, q_to: np.ndarray, vel_limit: float) -> List[tuple]:
        """关节空间 min-jerk 限速斜坡，返回 [(q, v), ...]。"""
        delta = q_to - q_from
        dist = float(np.max(np.abs(delta))) if delta.size else 0.0
        T = max(self._dt, dist / max(vel_limit, 1e-6))
        n = max(2, int(np.ceil(T / self._dt)) + 1)
        pts = []
        for i in range(n):
            tau = i / (n - 1)
            s = 10.0 * tau ** 3 - 15.0 * tau ** 4 + 6.0 * tau ** 5
            sd = (30.0 * tau ** 2 - 60.0 * tau ** 3 + 30.0 * tau ** 4) / T
            pts.append((q_from + s * delta, sd * delta))
        return pts

    def _consume_move_buffer(self) -> None:
        if self._move_idx < len(self._move_buffer):
            q, v = self._move_buffer[self._move_idx]
            self._move_idx += 1
            tau = self._gravity_tau(q)
            self._arm.mit(q, vel=v, kp=self._mit_kp, kd=self._mit_kd, tau=tau)
            if self._move_idx >= len(self._move_buffer):
                self._finish_move()
        else:
            self._finish_move()

    def _finish_move(self) -> None:
        q_end = self._move_buffer[-1][0] if self._move_buffer else self._ik_filter_pos
        q_end = np.asarray(q_end, dtype=np.float64).reshape(-1)[: self._n]
        self._ik_seed = q_end.copy()
        self._ik_filter_pos = q_end.copy()
        self._ik_filter_vel = np.zeros(self._n)
        self._ik_raw = q_end.copy()
        self._q_freeze = q_end.copy()
        self._seed_just_init = True
        self._consecutive_rejects = 0
        self._consecutive_ik_fails = 0
        self._sv = np.zeros(6)
        self._target_pose = self._fk_pose(q_end)
        self._prev_pose = None
        self._is_moving = False
        logger.info("已到达 %s", self._move_name)

    # ------------------------------------------------------------------
    # 模式切换：零力矩 / 急停
    # ------------------------------------------------------------------

    def _toggle_zero_torque(self) -> None:
        if self._is_moving:
            return
        new_state = not self._zero_torque
        logger.info("%s 零力矩模式...", "开启" if new_state else "关闭")
        self._zero_torque = new_state
        if new_state:
            print(">>> 零力矩模式已开启: 可手动拖动机械臂 <<<")
        else:
            self._resync_ik()
            print(">>> 零力矩模式已关闭: 恢复手柄控制 <<<")

    def _toggle_estop(self) -> None:
        new_state = not self._estop
        if new_state:
            # 冻结在当前控制目标位置（无重力下垂偏差）
            self._q_freeze = np.asarray(self._ik_filter_pos, dtype=np.float64).copy()
            self._estop = True
            print("\n!!! 急停已触发 — 位置锁定。再次按 Back 解除，或按 B/X 回 Home/零位 !!!")
        else:
            self._estop = False
            self._resync_ik()
            print(">>> 急停已解除: 恢复手柄控制 <<<")

    # ------------------------------------------------------------------
    # 夹爪
    # ------------------------------------------------------------------

    def _gripper_cmd(self, angle: float) -> None:
        if self._gripper is None:
            logger.info("夹爪未连接，忽略指令 (%.2f rad)", angle)
            return
        try:
            self._gripper.pos_vel(angle)
            logger.info("夹爪: %.2f rad", angle)
        except Exception as e:
            logger.error("夹爪控制异常: %s", e)

    # ------------------------------------------------------------------
    # Banner & 诊断
    # ------------------------------------------------------------------

    def _print_banner(self) -> None:
        print("\n" + "=" * 52)
        print("     reBotArm 手柄遥操作（增量笛卡尔控制）")
        print("=" * 52)
        print(f"  控制器映射:  {self._profile.display_name} [{self._profile.profile_id}]")
        print("  左摇杆       →  XY 平移")
        print("  LT / RT      →  Z 下/上")
        print("  右摇杆       →  Roll(Y) / Yaw(X)")
        print("  LB / RB      →  Pitch")
        print("  A            →  切换速度档")
        print("  B            →  回 Home")
        print("  X            →  回零位")
        print("  Y            →  零力矩模式（可拖动）")
        print("  D-pad ↑↓     →  夹爪 开/合" + ("" if self._gripper else "（未连接）"))
        print("  Back         →  急停 / 解除")
        print("  Start        →  退出")
        print("=" * 52)
        if self._target_pose is not None:
            p = self._target_pose
            print(f"  初始末端:    ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) m")
            print(f"               ({p[3]:.2f}, {p[4]:.2f}, {p[5]:.2f}) rad")
        self._log_speed()

    def _log_speed(self) -> None:
        name, factor = SPEED_LEVELS[self._speed_idx]
        lin_mm = self._max_lin_vel * factor * 1000
        ang = self._max_ang_vel * factor
        print(f"  速度档位:    {self._speed_idx + 1}/5 [{name}] "
              f"({lin_mm:.0f}mm/s, {ang:.2f}rad/s)")

    def _periodic_status(self) -> None:
        self._diag_tick += 1
        if self._diag_tick < int(self._rate * 5):
            return
        self._diag_tick = 0

        q = self._arm.get_positions(request=False)
        q = np.asarray(q, dtype=np.float64).reshape(-1)[: self._n]
        degs = [f"{v * 180 / math.pi:.1f}" for v in q]
        mode = "零力矩" if self._zero_torque else (
            "急停" if self._estop else ("运动中" if self._is_moving else "正常"))
        print(f"  [{mode}] 关节(°): [{', '.join(degs)}]")
        if self._target_pose is not None and not self._zero_torque:
            p = self._target_pose
            print(f"  末端目标: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) m  "
                  f"({p[3]:.2f}, {p[4]:.2f}, {p[5]:.2f}) rad")
