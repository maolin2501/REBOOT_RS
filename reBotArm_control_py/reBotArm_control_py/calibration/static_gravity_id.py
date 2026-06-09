"""程序 3：静态回归求质心（已知连杆质量）。

使用 Pinocchio ``computeJointTorqueRegressor``（v=0, a=0）在 6-DOF 简化模型上
对每连杆 10 参数中的 h = m·r 列做最小二乘。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pinocchio as pin

from .io import CALIBRATION_LINKS, TargetSample
from ..kinematics.robot_model import load_robot_model


def build_reduced_arm_model(urdf_path: Optional[str] = None) -> pin.Model:
    """锁定夹爪关节，得到 6-DOF 臂模型（nv=6）。"""
    model = load_robot_model(urdf_path=urdf_path)
    lock_ids = [
        model.getJointId("joint_left"),
        model.getJointId("joint_right"),
    ]
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


def build_gravity_regressor(
    model: pin.Model,
    data: pin.Data,
    q: np.ndarray,
) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(model.nq)
    v = np.zeros(model.nv)
    a = np.zeros(model.nv)
    pin.computeJointTorqueRegressor(model, data, q, v, a)
    return data.jointTorqueRegressor.copy()


def solve_com_from_measurements(
    records: List[TargetSample],
    known_masses: Dict[str, float],
    urdf_path: Optional[str] = None,
    exclude_joints: Optional[List[int]] = None,
    identify_joint_bias: bool = False,
    reg_lambda: float = 1e-3,
    bias_reg: float = 1e-4,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], dict]:
    """
    已知各 link 质量，最小二乘求质心（通过 h = m*r）。

    参数:
        exclude_joints: 不参与回归的关节索引(0-based)。例如竖直轴 joint1(=0)
            不承受重力，其力矩读数为纯摩擦/噪声，纳入会污染拟合，应剔除。
        identify_joint_bias: 是否为每个参与关节联合辨识一个常数力矩偏置 b_j。
            数据采集已用双向接近(plus/minus)平均消除库仑摩擦，剩余的方向无关
            恒定分量 = 重力 + 零位偏置；增设常数列把零位偏置与 h 分离，使 h
            更纯净，并把 b_j 输出供补偿端复现"静止保持力矩"。

    返回:
        coms: link_name -> com (3,)
        masses: 使用的质量 dict
        report: 残差、条件数、joint_bias 等
    """
    model = build_reduced_arm_model(urdf_path=urdf_path)
    data = model.createData()
    nv = model.nv
    n_param = (model.nbodies - 1) * 10

    excl = set(int(j) for j in (exclude_joints or []))
    keep_rows = [j for j in range(nv) if j not in excl]

    calib_bodies = {
        name: _link_to_body_id(model, name) for name in CALIBRATION_LINKS
    }

    phi_nom = _phi_from_inertia(model)
    for link, m in known_masses.items():
        if link in calib_bodies:
            body_id = calib_bodies[link]
            phi_nom[_body_param_base(body_id)] = m

    unknown_h_cols: List[int] = []
    known_m_cols: List[int] = []
    known_m_vals: List[float] = []

    for link in CALIBRATION_LINKS:
        if link not in known_masses:
            raise ValueError(f"缺少质量: {link}")
        body_id = calib_bodies[link]
        base = _body_param_base(body_id)
        known_m_cols.append(base)
        known_m_vals.append(known_masses[link])
        unknown_h_cols.extend([base + 1, base + 2, base + 3])

    unknown_set = set(unknown_h_cols)
    calib_m_set = set(known_m_cols)
    fixed_cols = [
        c
        for c in range(n_param)
        if c not in unknown_set and c not in calib_m_set
    ]
    m_known_vec = np.array(known_m_vals, dtype=float)

    Y_rows = []
    tau_rows = []

    for rec in records:
        q = np.asarray(rec.q_actual, dtype=float).reshape(model.nq)
        Y = build_gravity_regressor(model, data, q)
        tau_meas = np.asarray(rec.tau_static, dtype=float).reshape(nv)

        tau_fixed = (
            Y[:, fixed_cols] @ phi_nom[fixed_cols]
            + Y[:, known_m_cols] @ m_known_vec
        )

        Y_rows.append(Y[np.ix_(keep_rows, unknown_h_cols)])
        tau_rows.append((tau_meas - tau_fixed)[keep_rows])

    A = np.vstack(Y_rows)
    b = np.concatenate(tau_rows)

    h_nom = np.zeros(len(unknown_h_cols))
    for i, link in enumerate(CALIBRATION_LINKS):
        body_id = calib_bodies[link]
        h_nom[i * 3 : (i + 1) * 3] = (
            known_masses[link] * model.inertias[body_id].lever
        )

    n_h = len(unknown_h_cols)
    n_keep = len(keep_rows)
    num_poses = len(records)

    # 零位力矩偏置：为每个参与关节增设常数列(indicator)，与 h 联合辨识。
    # b 已按 (pose × keep_rows) 顺序堆叠，故 E = 纵向平铺单位阵。
    if identify_joint_bias:
        E = np.tile(np.eye(n_keep), (num_poses, 1))  # (num_poses*n_keep, n_keep)
        A_full = np.hstack([A, E])
    else:
        A_full = A

    # Tikhonov 正则化：h 拉向 URDF 名义质心；bias 轻正则拉向 0
    if identify_joint_bias:
        top = np.hstack([np.sqrt(reg_lambda) * np.eye(n_h), np.zeros((n_h, n_keep))])
        bot = np.hstack([np.zeros((n_keep, n_h)), np.sqrt(bias_reg) * np.eye(n_keep)])
        A_reg = np.vstack([A_full, top, bot])
        b_reg = np.concatenate([b, np.sqrt(reg_lambda) * h_nom, np.zeros(n_keep)])
    else:
        A_reg = np.vstack([A_full, np.sqrt(reg_lambda) * np.eye(n_h)])
        b_reg = np.concatenate([b, np.sqrt(reg_lambda) * h_nom])

    sol, residuals, rank, sv = np.linalg.lstsq(A_reg, b_reg, rcond=1e-8)
    cond_number = (
        float(sv[0] / sv[-1]) if len(sv) > 1 and sv[-1] > 1e-12 else float("inf")
    )

    h_sol = sol[:n_h]
    bias_keep = sol[n_h : n_h + n_keep] if identify_joint_bias else np.zeros(n_keep)
    joint_bias = np.zeros(nv)
    for k, j in enumerate(keep_rows):
        joint_bias[j] = bias_keep[k]

    tau_pred = A_full @ sol
    err = b - tau_pred
    rms = float(np.sqrt(np.mean(err**2)))

    coms: Dict[str, np.ndarray] = {}
    com_warnings: List[str] = []
    for i, link in enumerate(CALIBRATION_LINKS):
        m = known_masses[link]
        h = h_sol[i * 3 : (i + 1) * 3]
        coms[link] = h / m
        com_norm = float(np.linalg.norm(coms[link]))
        if not np.all(np.isfinite(coms[link])):
            raise ValueError(f"{link} 标定质心非有限值，数据异常")
        if com_norm > 0.35:
            # 静态重力仅辨识一阶矩 h=m·r（重力补偿用 h，拆分不影响补偿力矩）；
            # 质心偏大通常意味着给定质量偏小，提示但不中断。
            com_warnings.append(
                f"{link} 质心模长 {com_norm:.3f} m 偏大（疑似给定质量偏小，h=m·r 仍有效）"
            )

    phi_cal = phi_nom.copy()
    for i, col in enumerate(unknown_h_cols):
        phi_cal[col] = h_sol[i]

    # 逐位姿残差仅在参与回归的关节行上计算（与拟合一致）
    per_pose_residuals: List[float] = []
    for rec in records:
        q = np.asarray(rec.q_actual, dtype=float).reshape(model.nq)
        Y = build_gravity_regressor(model, data, q)
        tau_pred_pose = (Y @ phi_cal)[keep_rows] + bias_keep
        tau_meas_keep = np.asarray(rec.tau_static, dtype=float).reshape(nv)[keep_rows]
        per_pose_residuals.append(
            float(np.linalg.norm(tau_pred_pose - tau_meas_keep))
        )

    # err 按 (位姿 × 保留关节) 排列，映射回完整关节索引
    n_keep = len(keep_rows)
    err_mat = err.reshape(-1, n_keep) if n_keep and len(err) % n_keep == 0 else None
    per_joint_rms = np.zeros(nv)
    for k, j in enumerate(keep_rows):
        if err_mat is not None:
            per_joint_rms[j] = float(np.sqrt(np.mean(err_mat[:, k] ** 2)))

    report = {
        "rms_nm": rms,
        "cond_number": cond_number,
        "rank": int(rank),
        "num_samples": len(records),
        "num_unknowns": n_h + (n_keep if identify_joint_bias else 0),
        "excluded_joints": sorted(excl),
        "joint_bias": joint_bias.tolist(),
        "identify_joint_bias": bool(identify_joint_bias),
        "per_pose_residual_norm": per_pose_residuals,
        "per_joint_rms": per_joint_rms.tolist(),
        "com_warnings": com_warnings,
    }

    return coms, known_masses, report
