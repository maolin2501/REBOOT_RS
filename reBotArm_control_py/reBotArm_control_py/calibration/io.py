"""标定流程 YAML / NPZ 读写与路径工具。"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# 标定连杆（与 masses.yaml 一致）
CALIBRATION_LINKS = ("link1", "link2", "link3", "link4", "link5", "link6")
ARM_NUM_JOINTS = 6


def expand_arm_q(q_arm: np.ndarray, nq: int) -> np.ndarray:
    """6 维臂关节角 → 完整模型 q（夹爪关节置 0）。"""
    q = np.zeros(nq, dtype=float)
    q_arm = np.asarray(q_arm, dtype=float).reshape(ARM_NUM_JOINTS)
    q[: min(ARM_NUM_JOINTS, nq)] = q_arm[: min(ARM_NUM_JOINTS, nq)]
    return q


def arm_joint_slice(nv: int) -> slice:
    """实测力矩/回归矩阵使用的臂关节行索引。"""
    return slice(0, min(ARM_NUM_JOINTS, nv))


def default_config_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "config"


def default_urdf_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "urdf"
        / "00-arm-rs_asm-v3"
        / "urdf"
        / "00-arm-rs_asm-v3.urdf"
    )


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class SeedPose:
    name: str
    q: np.ndarray
    approach_delta: np.ndarray


@dataclass
class SeedConfig:
    urdf_path: Optional[str]
    home_q: np.ndarray
    segment_samples: int
    max_segment_dq: float
    limit_margin: float
    settle_time_s: float
    sample_time_s: float
    move_duration_s: float
    max_velocity_rad_s: float
    max_acceleration_rad_s2: float
    mit_kp: np.ndarray
    mit_kd: np.ndarray
    poses: List[SeedPose]


@dataclass
class Waypoint:
    name: str
    kind: str  # home | approach_minus | approach_plus | target | transit
    pose_name: str
    q: np.ndarray
    direction: Optional[str] = None  # "minus" | "plus" | None


@dataclass
class PoseReport:
    name: str
    accepted: bool
    reason: str = ""


@dataclass
class PathPlan:
    urdf_path: str
    urdf_sha256: str
    home_q: np.ndarray
    waypoints: List[Waypoint]
    pose_reports: List[PoseReport]
    segment_samples: int
    max_segment_dq: float
    limit_margin: float
    settle_time_s: float
    sample_time_s: float
    move_duration_s: float
    max_velocity_rad_s: float
    max_acceleration_rad_s2: float
    mit_kp: np.ndarray
    mit_kd: np.ndarray
    min_clearance_m: float = float("inf")
    total_path_dq: float = 0.0
    collision_check_enabled: bool = True


@dataclass
class TargetSample:
    pose_name: str
    q_target: np.ndarray
    q_actual: np.ndarray
    tau_static: np.ndarray
    tau_plus: np.ndarray
    tau_minus: np.ndarray
    tau_std: np.ndarray
    friction_delta: np.ndarray


def load_seed_config(path: str | Path) -> SeedConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    poses = []
    for p in raw.get("poses", []):
        poses.append(
            SeedPose(
                name=str(p["name"]),
                q=np.array(p["q"], dtype=float),
                approach_delta=np.array(p["approach_delta"], dtype=float),
            )
        )

    mit_kp = raw.get("mit_kp", [80.0] * 6)
    mit_kd = raw.get("mit_kd", [4.0] * 6)

    return SeedConfig(
        urdf_path=raw.get("urdf"),
        home_q=np.array(raw.get("home_q", [0.0] * 6), dtype=float),
        segment_samples=int(raw.get("segment_samples", 24)),
        max_segment_dq=float(raw.get("max_segment_dq", 0.35)),
        limit_margin=float(raw.get("limit_margin", 0.05)),
        settle_time_s=float(raw.get("settle_time_s", 0.5)),
        sample_time_s=float(raw.get("sample_time_s", 1.0)),
        move_duration_s=float(raw.get("move_duration_s", 2.0)),
        max_velocity_rad_s=float(raw.get("max_velocity_rad_s", 0.5)),
        max_acceleration_rad_s2=float(raw.get("max_acceleration_rad_s2", 1.5)),
        mit_kp=np.array(mit_kp, dtype=float),
        mit_kd=np.array(mit_kd, dtype=float),
        poses=poses,
    )


def load_masses(path: str | Path) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    masses = raw.get("masses_kg", raw)
    return {str(k): float(v) for k, v in masses.items()}


def _waypoint_to_dict(wp: Waypoint) -> dict:
    return {
        "name": wp.name,
        "kind": wp.kind,
        "pose_name": wp.pose_name,
        "q": wp.q.tolist(),
        "direction": wp.direction,
    }


def _waypoint_from_dict(d: dict) -> Waypoint:
    return Waypoint(
        name=str(d["name"]),
        kind=str(d["kind"]),
        pose_name=str(d["pose_name"]),
        q=np.array(d["q"], dtype=float),
        direction=d.get("direction"),
    )


def save_path(path: str | Path, plan: PathPlan) -> None:
    data = {
        "urdf_path": plan.urdf_path,
        "urdf_sha256": plan.urdf_sha256,
        "home_q": plan.home_q.tolist(),
        "segment_samples": plan.segment_samples,
        "max_segment_dq": plan.max_segment_dq,
        "limit_margin": plan.limit_margin,
        "settle_time_s": plan.settle_time_s,
        "sample_time_s": plan.sample_time_s,
        "move_duration_s": plan.move_duration_s,
        "max_velocity_rad_s": plan.max_velocity_rad_s,
        "max_acceleration_rad_s2": plan.max_acceleration_rad_s2,
        "mit_kp": plan.mit_kp.tolist(),
        "mit_kd": plan.mit_kd.tolist(),
        "min_clearance_m": plan.min_clearance_m,
        "total_path_dq": plan.total_path_dq,
        "collision_check_enabled": plan.collision_check_enabled,
        "pose_reports": [asdict(r) for r in plan.pose_reports],
        "waypoints": [_waypoint_to_dict(w) for w in plan.waypoints],
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def load_path(path: str | Path) -> PathPlan:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    reports = [
        PoseReport(
            name=str(r["name"]),
            accepted=bool(r["accepted"]),
            reason=str(r.get("reason", "")),
        )
        for r in raw.get("pose_reports", [])
    ]
    waypoints = [_waypoint_from_dict(w) for w in raw.get("waypoints", [])]

    return PathPlan(
        urdf_path=str(raw["urdf_path"]),
        urdf_sha256=str(raw["urdf_sha256"]),
        home_q=np.array(raw["home_q"], dtype=float),
        waypoints=waypoints,
        pose_reports=reports,
        segment_samples=int(raw.get("segment_samples", 24)),
        max_segment_dq=float(raw.get("max_segment_dq", 0.35)),
        limit_margin=float(raw.get("limit_margin", 0.05)),
        settle_time_s=float(raw.get("settle_time_s", 0.5)),
        sample_time_s=float(raw.get("sample_time_s", 1.0)),
        move_duration_s=float(raw.get("move_duration_s", 2.0)),
        max_velocity_rad_s=float(raw.get("max_velocity_rad_s", 0.5)),
        max_acceleration_rad_s2=float(raw.get("max_acceleration_rad_s2", 1.5)),
        mit_kp=np.array(raw.get("mit_kp", [80.0] * 6), dtype=float),
        mit_kd=np.array(raw.get("mit_kd", [4.0] * 6), dtype=float),
        min_clearance_m=float(raw.get("min_clearance_m", float("inf"))),
        total_path_dq=float(raw.get("total_path_dq", 0.0)),
        collision_check_enabled=bool(raw.get("collision_check_enabled", True)),
    )


def save_measurements(
    path: str | Path,
    records: List[TargetSample],
    meta: Dict[str, Any],
) -> None:
    payload: Dict[str, Any] = {"meta": meta, "records": []}
    for r in records:
        payload["records"].append(
            {
                "pose_name": r.pose_name,
                "q_target": r.q_target,
                "q_actual": r.q_actual,
                "tau_static": r.tau_static,
                "tau_plus": r.tau_plus,
                "tau_minus": r.tau_minus,
                "tau_std": r.tau_std,
                "friction_delta": r.friction_delta,
            }
        )
    np.savez_compressed(path, **{k: np.array(v) if isinstance(v, list) else v for k, v in _flatten_npz(payload).items()})


def _flatten_npz(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(_flatten_npz(v, key + "__"))
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                for i, item in enumerate(v):
                    out.update(_flatten_npz(item, f"{key}__{i}__"))
            else:
                out[key] = v
    return out


def load_measurements(path: str | Path) -> tuple[List[TargetSample], Dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    meta_raw = data.get("meta", None)
    if meta_raw is not None:
        meta = meta_raw.item() if hasattr(meta_raw, "item") else dict(meta_raw)
    else:
        meta = {}

    records: List[TargetSample] = []
    n = int(meta.get("num_records", 0))
    if n == 0:
        # 兼容扁平 npz
        i = 0
        while f"records__{i}__pose_name" in data.files:
            records.append(
                TargetSample(
                    pose_name=str(data[f"records__{i}__pose_name"].item()),
                    q_target=np.array(data[f"records__{i}__q_target"], dtype=float),
                    q_actual=np.array(data[f"records__{i}__q_actual"], dtype=float),
                    tau_static=np.array(data[f"records__{i}__tau_static"], dtype=float),
                    tau_plus=np.array(data[f"records__{i}__tau_plus"], dtype=float),
                    tau_minus=np.array(data[f"records__{i}__tau_minus"], dtype=float),
                    tau_std=np.array(data[f"records__{i}__tau_std"], dtype=float),
                    friction_delta=np.array(
                        data[f"records__{i}__friction_delta"], dtype=float
                    ),
                )
            )
            i += 1
        meta["num_records"] = len(records)
        return records, meta

    for i in range(n):
        prefix = f"records__{i}__"
        records.append(
            TargetSample(
                pose_name=str(data[f"{prefix}pose_name"].item()),
                q_target=np.array(data[f"{prefix}q_target"], dtype=float),
                q_actual=np.array(data[f"{prefix}q_actual"], dtype=float),
                tau_static=np.array(data[f"{prefix}tau_static"], dtype=float),
                tau_plus=np.array(data[f"{prefix}tau_plus"], dtype=float),
                tau_minus=np.array(data[f"{prefix}tau_minus"], dtype=float),
                tau_std=np.array(data[f"{prefix}tau_std"], dtype=float),
                friction_delta=np.array(data[f"{prefix}friction_delta"], dtype=float),
            )
        )
    return records, meta


def save_measurements_simple(
    path: str | Path,
    records: List[TargetSample],
    meta: Dict[str, Any],
) -> None:
    """使用 pickle 友好结构保存测量数据。"""
    import pickle

    blob = {
        "meta": meta,
        "records": [asdict(r) for r in records],
    }
    for i, r in enumerate(blob["records"]):
        for key in ("q_target", "q_actual", "tau_static", "tau_plus", "tau_minus", "tau_std", "friction_delta"):
            blob["records"][i][key] = np.asarray(blob["records"][i][key], dtype=float)

    with open(str(path).replace(".npz", ".pkl"), "wb") as f:
        pickle.dump(blob, f)

    # 同时写 npz（数组形式）
    meta_out = dict(meta)
    meta_out["num_records"] = len(records)
    arrays: Dict[str, Any] = {"meta": np.array(meta_out, dtype=object)}
    for i, r in enumerate(records):
        p = f"records__{i}__"
        arrays[f"{p}pose_name"] = np.array(r.pose_name)
        arrays[f"{p}q_target"] = r.q_target
        arrays[f"{p}q_actual"] = r.q_actual
        arrays[f"{p}tau_static"] = r.tau_static
        arrays[f"{p}tau_plus"] = r.tau_plus
        arrays[f"{p}tau_minus"] = r.tau_minus
        arrays[f"{p}tau_std"] = r.tau_std
        arrays[f"{p}friction_delta"] = r.friction_delta
    np.savez_compressed(path, **arrays)


# 对外统一使用 simple 版本
save_measurements = save_measurements_simple


def save_calibrated_inertia(
    path: str | Path,
    masses: Dict[str, float],
    coms: Dict[str, List[float]],
    meta: Dict[str, Any],
    joint_bias: Optional[List[float]] = None,
) -> None:
    inertia_params = {}
    for link in CALIBRATION_LINKS:
        if link not in masses or link not in coms:
            continue
        inertia_params[link] = {
            "mass": float(masses[link]),
            "com": [float(x) for x in coms[link]],
        }

    doc: Dict[str, Any] = {
        "use_calibrated_params": True,
        "inertia_params": inertia_params,
    }
    # 零位力矩偏置（每臂关节，N·m）：补偿端复现静止保持力矩用
    if joint_bias is not None:
        doc["joint_torque_bias"] = [float(x) for x in joint_bias]
    doc["meta"] = meta
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
