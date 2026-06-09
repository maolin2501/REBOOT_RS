"""ArmEndPos — 末端位置控制器（IK  + 轨迹规划二合一）。

统一的 posvel 控制器，同时支持两种运动模式：

  - ``move_to_ik(...)``   即时 IK 求解，关节角度一步到位（无轨迹平滑）。
  - ``move_to_traj(...)`` SE(3) 测地线规划 + CLIK 跟踪，末端沿平滑轨迹运动。

使用示例::

    arm = RobotArm()
    Arm_endpos_control = ArmEndPos(arm)
    Arm_endpos_control.start()

    # 即时 IK（适合近距离小幅运动）
    Arm_endpos_control.move_to_ik(x=0.3, y=0.0, z=0.3)

    # 带轨迹规划（平滑、可控时长）
    Arm_endpos_control.move_to_traj(x=0.3, y=0.0, z=0.3,
                        roll=0, pitch=0.4, yaw=0,
                        duration=2.0)

    Arm_endpos_control.end()

上下文管理器::

    with ArmEndPos(arm) as endpos:
        endpos.move_to_ik(x=0.3, y=0.0, z=0.3)
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import yaml

from ..kinematics import (
    compute_fk,
    pos_rot_to_se3,
    get_end_effector_frame_id,
    load_robot_model,
)
from ..kinematics.inverse_kinematics import (
    solve_ik,
    IKParams as TrajIKParams,
)
from ..trajectory import (
    TrajProfile,
    TrajPlanParams,
    IKParams as ClikIKParams,
    plan_cartesian_geodesic_trajectory,
    track_trajectory,
)
from ..dynamics import load_dynamics_model, compute_generalized_gravity
from ..actuator import RobotArm


def _inverse_output_torque(
    tau_des: np.ndarray, gain: np.ndarray, offset: np.ndarray,
) -> np.ndarray:
    """电机输出端逆变换：求指令使实际输出 = tau_des。

    电机输出模型 ``tau_actual = gain·cmd + offset·sign(cmd)``，
    其逆为 ``cmd = sign(τ)·max(0, |τ|−offset) / gain``。
    ``|τ| ≤ offset`` 落入死区（指令为 0），宁可轻微欠补偿也不过补偿。
    """
    s = np.sign(tau_des)
    mag = np.maximum(0.0, np.abs(tau_des) - offset) / gain
    return s * mag


class ArmEndPos:

    def __init__(
        self,
        arm: RobotArm,
        ctrl_rate: float = 200.0,
        profile: TrajProfile = TrajProfile.DOUBLE_S,
        feedforward_vel: bool = True,
        gravity_ff: bool = True,
        gravity_ff_ratio: float = 0.95,
        output_torque_calib: bool = True,
    ) -> None:
        """初始化控制器。

        参数:
            arm:              RobotArm 实例。
            ctrl_rate:        控制环频率（Hz），默认 200 Hz。轨迹按此频率插值，
                              并由控制环以同一频率逐点实时下发（无零阶保持错配）。
            profile:          轨迹时间轮廓类型，默认 DOUBLE_S（jerk 限制双 S 曲线）。
            feedforward_vel:  是否启用速度前馈。``True``（默认）→ 使用 MIT（电机
                              类型 1）模式，把规划关节速度作为 ``send_mit`` 的前馈
                              速度 ``vel`` 发送；``False`` → 沿用 POS_VEL（类型 2），
                              仅发送位置 + 速度上限，无前馈。
            gravity_ff:       是否启用重力前馈力矩补偿。``True``（默认）→ 按规划点
                              的重力补偿值 ``g(q)`` 作为 MIT 前馈力矩 ``tau`` 发送
                              （需 ``feedforward_vel=True`` 的 MIT 模式才生效）。
            gravity_ff_ratio: 重力前馈缩放系数，默认 0.95（轻微欠补偿更安全，
                              避免过补偿上抬）。
            output_torque_calib: 是否叠加电机输出端逆变换（默认 ``True``）。开启后
                              对重力前馈力矩做 ``cmd=sign(τ)·max(0,|τ|−b)/a`` 逆变换
                              （参数取自 ``config/motor_torque_calib.yaml``），抵消
                              电机增益+偏置，使实际输出力矩等于 0.95·g(q)。
        """
        self.arm = arm
        self._n = arm.num_joints
        self._ctrl_rate = float(ctrl_rate)
        self._dt = 1.0 / self._ctrl_rate
        self._feedforward_vel = bool(feedforward_vel)
        self._gravity_ff = bool(gravity_ff)
        self._gravity_ff_ratio = float(gravity_ff_ratio)
        self._output_torque_calib = bool(output_torque_calib)
        self._model = load_robot_model()
        self._end_frame_id = get_end_effector_frame_id(self._model)
        self._data = self._model.createData()

        # ── 重力补偿动力学模型（与零重力模式一致：优先标定惯量）──────────
        self._dyn_model = None
        self._dyn_data = None
        self._out_gain = np.ones(self._n)
        self._out_offset = np.zeros(self._n)
        if self._gravity_ff:
            cfg_dir = Path(__file__).resolve().parents[2] / "config"
            cal_yaml = cfg_dir / "inertia_calibrated.yaml"
            if cal_yaml.is_file():
                self._dyn_model = load_dynamics_model(calibration_yaml=str(cal_yaml))
                print(f"[ArmEndPos] 重力前馈: 标定惯量 {cal_yaml.name} ×{self._gravity_ff_ratio:.2f}")
            else:
                self._dyn_model = load_dynamics_model()
                print(f"[ArmEndPos] 重力前馈: URDF 名义惯量（未找到标定文件）×{self._gravity_ff_ratio:.2f}")
            self._dyn_data = self._dyn_model.createData()

            if self._output_torque_calib:
                self._out_gain, self._out_offset = self._load_output_torque_calib(
                    cfg_dir / "motor_torque_calib.yaml"
                )
                print("[ArmEndPos] 输出端逆变换 cmd=sign(τ)·max(0,|τ|−b)/a:")
                for i, jc in enumerate(arm._joints):
                    print(f"  {jc.name}({jc.model}): a={self._out_gain[i]:.4f} "
                          f"b={self._out_offset[i]:+.4f}")

        self._pv_vlim = np.array([j.vlim for j in arm._joints], dtype=np.float64)
        # MIT 模式增益（前馈速度经 kd 项生效：τ=kp·Δp + kd·(v_ff−v) + tau）
        self._mit_kp = np.array([j.kp for j in arm._joints], dtype=np.float64)
        self._mit_kd = np.array([j.kd for j in arm._joints], dtype=np.float64)

        self._traj_params = TrajPlanParams(dt=self._dt, profile=profile)
        self._ik_solver_params = TrajIKParams(
            max_iter=200, tolerance=1e-4, step_size=0.5, damping=1e-6,
        )
        self._clik_params = ClikIKParams(
            max_iter=200, tolerance=1e-4, damping=1e-6, step_size=0.8,
        )

        self._q_target = np.zeros(self._n)
        self._v_target = np.zeros(self._n)        # 前馈关节速度（保持时为 0）
        self._tau_target = np.zeros(self._n)      # 重力前馈力矩（保持时维持）
        self._running = False

        # ── 轨迹回放状态（由 200 Hz 控制环逐点消费） ─────────────────────
        # 每个元素为 (q, v, tau)：关节位置、前馈速度、重力前馈力矩。
        self._traj_lock = threading.Lock()
        self._traj: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._traj_idx = 0
        self._moving = False

        # ── 回零安全参数 ──────────────────────────────────────────────────
        self._home_vel: float = 0.3   # rad/s，回零限速
        self._vlim_override: np.ndarray | None = None   # safe_home 期间生效

    # ── 生命周期 ───────────────────────────────────────────────────────────

    def start(self) -> None:
        """连接、切换模式、使能、以 ctrl_rate（默认 200 Hz）启动控制循环。

        启用前馈时切换到 MIT（电机类型 1）模式，否则用 POS_VEL（类型 2）。
        """
        self.arm.connect()
        if self._feedforward_vel:
            self.arm.mode_mit(kp=self._mit_kp, kd=self._mit_kd)
        else:
            self.arm.mode_pos_vel()
        self.arm.enable()
        self.arm.start_control_loop(self._loop_cb, rate=self._ctrl_rate)
        self._running = True

    def end(self) -> None:
        """安全回零后再断开连接。"""
        if not self._running:
            return
        self.safe_home()
        self.arm.disconnect()
        self._running = False

    def __enter__(self) -> "ArmEndPos":
        return self

    def __exit__(self, *args) -> None:
        self.end()

    # ── 重力前馈 ───────────────────────────────────────────────────────────

    def _load_output_torque_calib(
        self, calib_yaml: Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        """按电机型号读取输出端 (gain, offset)，逐臂关节返回。"""
        gain = np.ones(self._n)
        offset = np.zeros(self._n)
        if not calib_yaml.is_file():
            print(f"[ArmEndPos] 未找到 {calib_yaml.name}，输出端逆变换按恒等处理")
            return gain, offset
        with open(calib_yaml, "r", encoding="utf-8") as f:
            models = (yaml.safe_load(f) or {}).get("models", {})
        for i, jc in enumerate(self.arm._joints):
            mc = models.get(jc.model, {})
            gain[i] = float(mc.get("gain", 1.0))
            offset[i] = float(mc.get("offset", 0.0))
        return gain, offset

    def _gravity_tau(self, q: np.ndarray) -> np.ndarray:
        """规划点重力补偿前馈力矩 = 输出端逆变换(ratio · g(q))。

        重力补偿值 g(q) 取自标定动力学模型（与零重力模式一致），
        缩放 ``gravity_ff_ratio``（默认 0.95，轻微欠补偿更安全），
        再对臂关节施加电机输出端逆变换，使实际输出力矩 ≈ 0.95·g(q)。
        未启用重力前馈时返回零向量。
        """
        if not self._gravity_ff or self._dyn_model is None:
            return np.zeros(np.asarray(q).shape[-1])
        q_full = np.asarray(q, dtype=np.float64).reshape(-1)
        if q_full.shape[0] != self._dyn_model.nq:
            qf = np.zeros(self._dyn_model.nq)
            k = min(self._dyn_model.nq, q_full.shape[0])
            qf[:k] = q_full[:k]
            q_full = qf
        g = compute_generalized_gravity(
            model=self._dyn_model, q=q_full, data=self._dyn_data,
        )
        tau = self._gravity_ff_ratio * g
        if self._output_torque_calib:
            n = min(self._n, tau.shape[0])
            tau[:n] = _inverse_output_torque(
                tau[:n], self._out_gain[:n], self._out_offset[:n],
            )
        return tau

    # ── 公共 API ───────────────────────────────────────────────────────────

    def safe_home(self, vlim: float | None = None) -> None:
        """以安全速度返回零位。

        关节空间限速斜坡（min-jerk），载入轨迹缓冲交由控制环逐点回放，
        因此对 MIT 与 POS_VEL 两种模式同样限速安全：MIT 模式下前馈速度
        随斜坡给出，避免位置阶跃造成的急动。
        """
        if not self._running:
            return
        v = self._home_vel if vlim is None else float(vlim)
        self._vlim_override = np.full(self._n, v, dtype=np.float64)

        q_curr, _, _ = self.arm.get_state()
        q_curr = np.asarray(q_curr, dtype=np.float64).reshape(-1)
        self._load_joint_ramp(q_curr, np.zeros_like(q_curr), v)

        deadline = time.monotonic() + 30.0
        while True:
            with self._traj_lock:
                done = not self._moving
            q, _, _ = self.arm.get_state()
            if done and np.max(np.abs(q)) < 0.01:
                break
            if time.monotonic() > deadline:
                print("[ArmEndPos] safe_home 超时")
                break
            time.sleep(self._dt)
        self._vlim_override = None

    def _load_joint_ramp(
        self, q_from: np.ndarray, q_to: np.ndarray, vel_limit: float,
    ) -> None:
        """生成关节空间 min-jerk 限速斜坡并载入轨迹缓冲。"""
        delta = q_to - q_from
        dist = float(np.max(np.abs(delta)))
        T = max(self._dt, dist / max(vel_limit, 1e-6))
        n = max(2, int(np.ceil(T / self._dt)) + 1)
        dt = T / (n - 1)
        pts = []
        for i in range(n):
            tau = i / (n - 1)
            s = 10.0 * tau ** 3 - 15.0 * tau ** 4 + 6.0 * tau ** 5      # 位置剖面
            sd = (30.0 * tau ** 2 - 60.0 * tau ** 3 + 30.0 * tau ** 4) / T  # 速度剖面
            q_i = q_from + s * delta
            pts.append((q_i, sd * delta, self._gravity_tau(q_i)))
        with self._traj_lock:
            self._traj = pts
            self._traj_idx = 0
            self._moving = True

    def move_to_ik(
        self,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ) -> bool:
        """IK 求解并驱动机械臂移动到目标位姿（无轨迹平滑）。

        参数:
            x, y, z:        目标末端位置（米）。
            roll, pitch, yaw: 目标姿态欧拉角（弧度），默认零姿态。

        返回:
            IK 求解成功返回 ``True``，否则 ``False``。
        """
        if not self._running:
            return False

        q_curr, _, _ = self.arm.get_state()
        T_target = pos_rot_to_se3(np.array([x, y, z]), roll=roll, pitch=pitch, yaw=yaw)

        result = solve_ik(
            self._model, self._data, self._end_frame_id,
            T_target, q_curr, self._ik_solver_params,
        )
        if not result.success:
            print(f"[ArmEndPos/IK] IK 未收敛  err={result.error:.3e}")
            return False

        tau_g = self._gravity_tau(result.q)
        with self._traj_lock:
            self._moving = False
            self._traj = []
            self._traj_idx = 0
            self._q_target = result.q.copy()
            self._v_target = np.zeros_like(result.q)
            self._tau_target = tau_g
        return True

    def move_to_traj(
        self,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        duration: float = 2.0,
    ) -> bool:
        """SE(3) 测地线规划 + CLIK 跟踪，驱动机械臂沿平滑轨迹运动。

        参数:
            x, y, z:        目标末端位置（米）。
            roll, pitch, yaw: 目标姿态欧拉角（弧度），默认零姿态。
            duration:       运动时长（秒）。若 ``<= 0`` 则根据末端移动距离自动估算。

        返回:
            规划与求解成功返回 ``True``，否则 ``False``。
        """
        if not self._running:
            return False

        q_start, _, _ = self.arm.get_state()

        T_target = pos_rot_to_se3(
            np.array([x, y, z]), roll=roll, pitch=pitch, yaw=yaw,
        )

        ik_result = solve_ik(
            self._model, self._data, self._end_frame_id,
            T_target, q_start, self._ik_solver_params,
        )
        if not ik_result.success:
            print(f"[ArmEndPos/Traj] IK 失败  err={ik_result.error:.4f}")
            return False

        q_end = ik_result.q

        T_start = compute_fk(self._model, q_start)[2]
        T_end = compute_fk(self._model, q_end)[2]

        if duration <= 0:
            dist = float(np.linalg.norm(T_target.translation() - T_start.translation()))
            duration = max(1.0, dist / 0.1)

        cart_traj = plan_cartesian_geodesic_trajectory(
            T_start, T_end, duration, self._traj_params,
        )

        joint_traj = track_trajectory(
            self._model, self._end_frame_id,
            cart_traj.trajectory, q_start, self._clik_params,
            null_gain=0.1,
        )
        if not joint_traj:
            print("[ArmEndPos/Traj] 轨迹为空")
            return False

        # (q, v, tau)：关节位置 + 规划前馈速度（双 S 端点速度为 0）+ 规划点重力前馈
        pts = [
            (pt.q.copy(),
             pt.v.copy() if pt.v is not None else np.zeros(self._model.nv),
             self._gravity_tau(pt.q))
            for pt in joint_traj
        ]

        # 装载回放缓冲，交由 200 Hz 控制环逐点实时下发
        with self._traj_lock:
            self._traj = pts
            self._traj_idx = 0
            self._moving = True
        return True

    # ── 控制循环（ctrl_rate，默认 200 Hz）─────────────────────────────────

    def _loop_cb(self, _: RobotArm, dt: float) -> None:
        """每个控制周期推进一个轨迹点并下发；无轨迹时保持当前位置（前馈速度 0）。

        插值点已按 ctrl_rate 采样，故每 tick 恰好消费一点，回放时长与规划
        时长一致，且下发频率 == 插值频率 == ctrl_rate（无零阶保持错配）。
        启用前馈时经 MIT（类型 1）下发位置 + 规划速度 vel；否则 POS_VEL 仅位置。
        """
        with self._traj_lock:
            if self._moving and self._traj_idx < len(self._traj):
                self._q_target, self._v_target, self._tau_target = \
                    self._traj[self._traj_idx]
                self._traj_idx += 1
                if self._traj_idx >= len(self._traj):
                    self._moving = False
            else:
                self._v_target = np.zeros_like(self._q_target)
                # 保持态：维持当前位姿的重力补偿，避免松弛下沉
            target = self._q_target
            vff = self._v_target
            tau_ff = self._tau_target

        if self._feedforward_vel:
            self.arm.mit(target, vel=vff, kp=self._mit_kp, kd=self._mit_kd, tau=tau_ff)
        else:
            vlim = (self._vlim_override
                    if self._vlim_override is not None else self._pv_vlim)
            self.arm.pos_vel(target, vlim=vlim)
