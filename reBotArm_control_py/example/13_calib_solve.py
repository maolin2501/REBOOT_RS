#!/usr/bin/env python3
"""程序 3：标定参数求解（离线）。

从测量数据 + 实测质量，最小二乘求解各连杆质心，写出 inertia_calibrated.yaml。

用法:
    uv run python example/13_calib_solve.py
    uv run python example/13_calib_solve.py --measurements config/calibration_measurements.npz
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reBotArm_control_py.calibration import (
    CALIBRATION_LINKS,
    build_pose_validator,
    default_config_dir,
    default_urdf_path,
    load_measurements,
    load_masses,
    save_calibrated_inertia,
    solve_com_from_measurements,
)
from reBotArm_control_py.calibration.static_gravity_id import build_reduced_arm_model


def _nominal_com(model: pin.Model, link: str) -> np.ndarray:
    body_id = model.getFrameId(link)
    jid = int(model.frames[body_id].parentJoint)
    return model.inertias[jid].lever.copy()


def _load_joint_models(arm_yaml: Path) -> list:
    """从 arm.yaml 读取各关节电机型号，按关节顺序返回。"""
    with open(arm_yaml, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return [str(j.get("model", "")) for j in raw.get("joints", [])]


def _build_torque_calib(arm_yaml: Path, calib_yaml: Path):
    """构造逐关节 (gain, offset) 数组。返回 (gain[nv], offset[nv], 描述)。"""
    models = _load_joint_models(arm_yaml)
    with open(calib_yaml, "r", encoding="utf-8") as f:
        cfg = (yaml.safe_load(f) or {}).get("models", {})
    nj = len(models)
    gain = np.ones(nj)
    offset = np.zeros(nj)
    desc = []
    for i, m in enumerate(models):
        mc = cfg.get(m, {})
        gain[i] = float(mc.get("gain", 1.0))
        offset[i] = float(mc.get("offset", 0.0))
        desc.append(f"joint{i+1}({m}): a={gain[i]:.4f} b={offset[i]:+.4f}")
    return gain, offset, desc


def _apply_torque_calib(records, gain, offset):
    """对每条记录的实测力矩做 tau_actual = a*tau + b*sign(tau)，保号奇对称。"""
    nv = len(gain)
    for r in records:
        for arr_name in ("tau_static", "tau_plus", "tau_minus"):
            tau = np.asarray(getattr(r, arr_name), dtype=float)
            n = min(nv, tau.shape[0])
            tau[:n] = gain[:n] * tau[:n] + offset[:n] * np.sign(tau[:n])
            setattr(r, arr_name, tau)
    return records


def _clean_records(records, masses, urdf_path, reject_sigma, drop_ground,
                   exclude_joints=None, identify_joint_bias=False):
    """剔除异常点：(1) 实际位姿触地/自碰撞；(2) 残差离群（疑似电机失能）。

    返回 (保留记录, 剔除信息列表)。
    """
    dropped = []
    kept = list(records)

    # (1) 实际位姿干涉检查（含水平面）
    if drop_ground:
        validator = build_pose_validator(enable_collision=True, enable_ground=True)
        survivors = []
        for r in kept:
            ok, reason, _ = validator.check_static(np.asarray(r.q_actual))
            if ok:
                survivors.append(r)
            else:
                dropped.append((r.pose_name, f"实际位姿干涉: {reason}"))
        kept = survivors

    # (2) 残差离群剔除（基于中位数 + sigma·MAD，鲁棒）
    for _ in range(3):
        if len(kept) < 8:
            break
        _, _, rep = solve_com_from_measurements(
            kept, masses, urdf_path=urdf_path, exclude_joints=exclude_joints,
            identify_joint_bias=identify_joint_bias,
        )
        res = np.array(rep["per_pose_residual_norm"])
        med = float(np.median(res))
        mad = float(np.median(np.abs(res - med))) + 1e-9
        thresh = med + reject_sigma * 1.4826 * mad
        keep_mask = res <= thresh
        if keep_mask.all():
            break
        for idx, (r, keep) in enumerate(zip(kept, keep_mask)):
            if not keep:
                dropped.append(
                    (r.pose_name, f"残差离群 {res[idx]:.3f}>阈值{thresh:.3f}(疑似失能)")
                )
        kept = [r for r, keep in zip(kept, keep_mask) if keep]

    return kept, dropped


def main() -> int:
    parser = argparse.ArgumentParser(description="重力补偿标定 — 程序3 参数求解")
    parser.add_argument(
        "--measurements",
        type=str,
        default=str(default_config_dir() / "calibration_measurements.npz"),
    )
    parser.add_argument(
        "--masses",
        type=str,
        default=str(default_config_dir() / "calibration_masses.yaml"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(default_config_dir() / "inertia_calibrated.yaml"),
    )
    parser.add_argument("--urdf", type=str, default=None)
    parser.add_argument("--reject-sigma", type=float, default=3.0,
                        help="残差离群剔除阈值(中位数+sigma·MAD)，设 0 关闭")
    parser.add_argument("--no-clean", action="store_true", help="不做异常点剔除")
    parser.add_argument("--torque-calib", type=str,
                        default=str(default_config_dir() / "motor_torque_calib.yaml"),
                        help="电机扭矩标定文件(set->actual)；不存在则跳过")
    parser.add_argument("--no-torque-calib", action="store_true",
                        help="不做电机扭矩修正")
    parser.add_argument("--arm-yaml", type=str,
                        default=str(default_config_dir() / "arm.yaml"))
    parser.add_argument("--exclude-joints", type=str, default="1",
                        help="不参与回归的关节(1-based,逗号分隔)。默认 1 "
                             "(竖直轴 joint1 不承重，纯摩擦噪声)。设空字符串关闭")
    parser.add_argument("--no-joint-bias", action="store_true",
                        help="不辨识各关节零位力矩偏置(默认辨识，基于重力对称)")
    args = parser.parse_args()

    exclude_joints = [int(s) - 1 for s in args.exclude_joints.split(",") if s.strip()]
    identify_bias = not args.no_joint_bias

    records, meta = load_measurements(args.measurements)
    if len(records) < 6:
        print(f"[错误] 测量样本过少 ({len(records)}), 建议 ≥ 12 个位姿")
        return 1

    masses = load_masses(args.masses)
    urdf_path = args.urdf or meta.get("urdf_path") or str(default_urdf_path())

    print("=" * 60)
    print("  程序 3：标定参数求解（固定质量，仅求质心）")
    print(f"  原始样本数: {len(records)}")
    print(f"  URDF: {urdf_path}")
    print("=" * 60)

    # ---- 电机扭矩修正：把反馈/设定扭矩换算成实际输出扭矩 ----
    torque_calib_info = None
    if not args.no_torque_calib and Path(args.torque_calib).is_file():
        gain, offset, desc = _build_torque_calib(Path(args.arm_yaml), Path(args.torque_calib))
        records = _apply_torque_calib(records, gain, offset)
        torque_calib_info = {"gain": gain.tolist(), "offset": offset.tolist()}
        print("\n[电机扭矩修正] tau_actual = a·tau + b·sign(tau)")
        for d in desc:
            print(f"  {d}")

    if exclude_joints:
        print("\n[排除关节] 不参与回归: "
              + ", ".join(f"joint{j+1}" for j in exclude_joints)
              + "（竖直/不承重轴，读数为摩擦噪声）")

    n_raw = len(records)
    if not args.no_clean:
        records, dropped = _clean_records(
            records, masses, urdf_path,
            reject_sigma=args.reject_sigma, drop_ground=True,
            exclude_joints=exclude_joints, identify_joint_bias=identify_bias,
        )
        print(f"\n[异常点剔除] 原始 {n_raw} → 保留 {len(records)}，剔除 {len(dropped)}")
        for name, why in dropped:
            print(f"  - {name:22s} {why}")
        if len(records) < 6:
            print("[错误] 清洗后样本过少，无法求解")
            return 1

    coms, masses_used, report = solve_com_from_measurements(
        records, masses, urdf_path=urdf_path, exclude_joints=exclude_joints,
        identify_joint_bias=identify_bias,
    )

    model = build_reduced_arm_model(urdf_path=urdf_path)

    print("\n[求解质量]")
    print(f"  最终样本:     {report['num_samples']}")
    print(f"  残差 RMS:     {report['rms_nm']:.4f} N·m  (目标 < 0.2)")
    print(f"  条件数 cond:  {report['cond_number']:.2f}")
    print(f"  秩 rank:      {report['rank']} / {report['num_unknowns']}"
          f"  ({'满秩，质心全可辨识' if report['rank']==report['num_unknowns'] else '秩亏，激励不足'})")
    for w in report.get("com_warnings", []):
        print(f"  [注意] {w}")

    joint_bias = np.array(report.get("joint_bias", []), dtype=float)
    if identify_bias and joint_bias.size:
        print("  [零位偏置] "
              + " ".join(f"j{j+1}={joint_bias[j]:+.3f}" for j in range(joint_bias.size))
              + " N·m  (基于重力对称分离的恒定零位)")

    # ---- 有效性评估：URDF 名义 vs 标定 对实测力矩的预测对比 ----
    from reBotArm_control_py.calibration.static_gravity_id import (
        build_gravity_regressor, _phi_from_inertia, _link_to_body_id, _body_param_base,
    )
    dyn = build_reduced_arm_model(urdf_path=urdf_path)
    ddata = dyn.createData()
    phi_nom = _phi_from_inertia(dyn)
    phi_cal = phi_nom.copy()
    for link in CALIBRATION_LINKS:
        base = _body_param_base(_link_to_body_id(dyn, link))
        h = coms[link] * masses_used[link]
        phi_cal[base + 1: base + 4] = h
    nv = dyn.nv
    err_nom = []
    err_cal = []
    for rec in records:
        q = np.asarray(rec.q_actual, dtype=float).reshape(dyn.nq)
        Y = build_gravity_regressor(dyn, ddata, q)
        tau_m = np.asarray(rec.tau_static, dtype=float).reshape(nv)
        bias_full = joint_bias if joint_bias.size == nv else np.zeros(nv)
        err_nom.append(Y @ phi_nom - tau_m)
        err_cal.append(Y @ phi_cal + bias_full - tau_m)
    err_nom = np.array(err_nom)
    err_cal = np.array(err_cal)
    incl = [j for j in range(nv) if j not in set(exclude_joints)]
    rms_nom = float(np.sqrt(np.mean(err_nom[:, incl] ** 2)))
    rms_cal = float(np.sqrt(np.mean(err_cal[:, incl] ** 2)))
    pj_nom = np.sqrt(np.mean(err_nom ** 2, axis=0))
    pj_cal = np.sqrt(np.mean(err_cal ** 2, axis=0))
    print("\n[有效性评估] 重力力矩预测误差 RMS：URDF 名义 → 标定 (N·m)")
    print(f"  {'joint':8s} {'名义RMS':>9s} {'标定RMS':>9s} {'改善%':>8s}")
    for j in range(nv):
        impr = (1 - pj_cal[j] / pj_nom[j]) * 100 if pj_nom[j] > 1e-9 else 0.0
        tag = "  (排除)" if j in set(exclude_joints) else ""
        print(f"  joint{j+1:<3d} {pj_nom[j]:9.3f} {pj_cal[j]:9.3f} {impr:8.1f}{tag}")
    overall = (1 - rms_cal / rms_nom) * 100 if rms_nom > 1e-9 else 0.0
    print(f"  {'总体*':8s} {rms_nom:9.3f} {rms_cal:9.3f} {overall:8.1f}  (*仅含参与回归关节)")
    report["rms_nominal_nm"] = rms_nom
    report["rms_calibrated_nm"] = rms_cal
    report["improvement_pct"] = overall

    print("\n[连杆质心] URDF 名义 vs 标定 (mm)")
    print(f"  {'link':8s}  {'m(kg)':8s}  {'nom_x':>8s} {'nom_y':>8s} {'nom_z':>8s}  "
          f"{'cal_x':>8s} {'cal_y':>8s} {'cal_z':>8s}  {'d(mm)':>8s}")
    print("-" * 72)
    for link in CALIBRATION_LINKS:
        nom = _nominal_com(model, link)
        cal = coms[link]
        delta_mm = np.linalg.norm(cal - nom) * 1000.0
        print(
            f"  {link:8s}  {masses_used[link]:8.4f}  "
            f"{nom[0]*1000:8.1f} {nom[1]*1000:8.1f} {nom[2]*1000:8.1f}  "
            f"{cal[0]*1000:8.1f} {cal[1]*1000:8.1f} {cal[2]*1000:8.1f}  "
            f"{delta_mm:8.1f}"
        )

    print("\n[逐位姿残差] (N·m)")
    for rec, res in zip(records, report["per_pose_residual_norm"]):
        print(f"  {rec.pose_name:22s}  {res:.4f}")

    if report["rms_nm"] > 0.2:
        print("\n[建议] 残差偏大：检查实测质量、增加激励位姿后重跑程序 1+2")
    if report["cond_number"] > 100:
        print("[建议] 条件数偏大：位姿分布不足，请增加差异化构型")

    save_meta = {
        **meta,
        "solved_at": datetime.now(timezone.utc).isoformat(),
        "solver_report": report,
        "masses_yaml": str(args.masses),
        "measurements_file": str(args.measurements),
        "torque_calib": torque_calib_info,
    }
    coms_list = {k: v.tolist() for k, v in coms.items()}
    save_calibrated_inertia(
        args.output, masses_used, coms_list, save_meta,
        joint_bias=report.get("joint_bias") if identify_bias else None,
    )
    print(f"\n[完成] 已写入: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
