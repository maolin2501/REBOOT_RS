"""关节限位与 URDF 自碰撞检验（可选 HPP-FCL）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pinocchio as pin

from .io import ARM_NUM_JOINTS, default_urdf_path, expand_arm_q


@dataclass
class PoseValidator:
    model: pin.Model
    data: pin.Data
    collision_enabled: bool
    geom_model: Optional[pin.GeometryModel]
    geom_data: Optional[pin.GeometryData]
    limit_margin: float

    def check_limits(self, q: np.ndarray) -> Tuple[bool, str]:
        q = expand_arm_q(q, self.model.nq)
        lo = self.model.lowerPositionLimit + self.limit_margin
        hi = self.model.upperPositionLimit - self.limit_margin
        for i in range(ARM_NUM_JOINTS):
            if q[i] < lo[i] - 1e-9:
                return False, f"joint{i+1} 低于下限: {q[i]:.4f} < {lo[i]:.4f}"
            if q[i] > hi[i] + 1e-9:
                return False, f"joint{i+1} 超过上限: {q[i]:.4f} > {hi[i]:.4f}"
        return True, ""

    def check_collision(
        self, q: np.ndarray, with_clearance: bool = True
    ) -> Tuple[bool, str, float]:
        """自碰撞布尔判断；with_clearance=True 时额外计算最小间隙（较慢）。"""
        if not self.collision_enabled or self.geom_model is None:
            return True, "", float("inf")

        q = expand_arm_q(q, self.model.nq)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateGeometryPlacements(
            self.model, self.data, self.geom_model, self.geom_data, q
        )

        # stopAtFirstCollision=True：命中即返回，布尔判断快
        pin.computeCollisions(
            self.model,
            self.data,
            self.geom_model,
            self.geom_data,
            q,
            not with_clearance,
        )
        for k, pair in enumerate(self.geom_model.collisionPairs):
            res = self.geom_data.collisionResults[k]
            if res.isCollision():
                g1 = self.geom_model.geometryObjects[pair.first].name
                g2 = self.geom_model.geometryObjects[pair.second].name
                return False, f"自碰撞: {g1} <-> {g2}", 0.0

        if not with_clearance:
            return True, "", float("inf")

        min_dist = float("inf")
        try:
            pin.computeDistances(
                self.model,
                self.data,
                self.geom_model,
                self.geom_data,
                q,
            )
            for res in self.geom_data.distanceResults:
                if res.min_distance < min_dist:
                    min_dist = float(res.min_distance)
        except Exception:
            min_dist = float("inf")

        return True, "", min_dist

    def check_static(
        self, q: np.ndarray, with_clearance: bool = True
    ) -> Tuple[bool, str, float]:
        ok, reason = self.check_limits(q)
        if not ok:
            return False, reason, 0.0
        ok, reason, min_d = self.check_collision(q, with_clearance=with_clearance)
        if not ok:
            return False, reason, min_d
        return True, "", min_d

    def check_segment(
        self,
        q_a: np.ndarray,
        q_b: np.ndarray,
        n_samples: int = 20,
    ) -> Tuple[bool, float, str, float]:
        """在 q_a→q_b 间线性插值检验。返回 (ok, first_bad_alpha, reason, min_clearance)。

        段内插值仅做快速布尔自碰撞；最小间隙仅在两端点计算以控制耗时。
        """
        q_a = np.asarray(q_a, dtype=float)
        q_b = np.asarray(q_b, dtype=float)
        min_clear = float("inf")
        for k in range(n_samples + 1):
            alpha = k / max(n_samples, 1)
            q = (1.0 - alpha) * q_a + alpha * q_b
            endpoint = k == 0 or k == n_samples
            ok, reason, md = self.check_static(q, with_clearance=endpoint)
            if endpoint:
                min_clear = min(min_clear, md)
            if not ok:
                return False, alpha, reason, min_clear
        return True, 1.0, "", min_clear


def _add_ground_plane(
    geom_model: pin.GeometryModel,
    ground_z: float = 0.0,
    exclude_links: Tuple[str, ...] = ("base_link",),
) -> Optional[int]:
    """向几何模型加入水平地面（半空间 z<=ground_z）并与各连杆建立碰撞对。

    连杆任意部分低于该平面即判为与水平面干涉。base 安装面在 z=0，默认排除。
    返回地面几何体索引；若 hppfcl 不可用返回 None。
    """
    try:
        import hppfcl as fcl
    except Exception:
        try:
            import coal as fcl  # 新版命名
        except Exception:
            return None

    # 记录加入地面前的连杆几何体索引
    link_indices = list(range(len(geom_model.geometryObjects)))

    # 用大平板 Box 表示地面：顶面位于 z=ground_z，向下 0.5m 厚。
    # （hppfcl 的 Halfspace 与网格碰撞极慢，Box-网格快约 3 个数量级。）
    half_thick = 0.5
    plate = 6.0
    ground_geom = fcl.Box(plate, plate, half_thick)
    placement = pin.SE3(np.eye(3), np.array([0.0, 0.0, float(ground_z) - half_thick / 2.0]))
    go = pin.GeometryObject("ground_plane", 0, placement, ground_geom)
    ground_idx = geom_model.addGeometryObject(go)

    def _excluded(name: str) -> bool:
        for ex in exclude_links:
            if name == ex or name.startswith(ex + "_"):
                return True
        return False

    for li in link_indices:
        name = geom_model.geometryObjects[li].name
        if _excluded(name):
            continue
        geom_model.addCollisionPair(pin.CollisionPair(ground_idx, li))
    return int(ground_idx)


def _remove_adjacent_collision_pairs(
    model: pin.Model, geom_model: pin.GeometryModel
) -> None:
    """排除 URDF 父子相邻 link 的碰撞对，降低 mesh 接缝误报。"""
    # 标定全程锁定的夹爪关节：二者相对位姿固定，其互检为装配固定贴合，非运动风险
    locked_gripper = set()
    for jname in ("joint_left", "joint_right"):
        if model.existJointName(jname):
            locked_gripper.add(int(model.getJointId(jname)))

    to_remove: List[int] = []
    njoints = len(model.parents)
    for idx, pair in enumerate(geom_model.collisionPairs):
        go1 = geom_model.geometryObjects[pair.first]
        go2 = geom_model.geometryObjects[pair.second]
        # 同一 body（同 parentJoint）或直接父子关节，均视为机械相邻，排除误报
        j1 = int(go1.parentJoint)
        j2 = int(go2.parentJoint)
        if j1 == j2:
            to_remove.append(idx)
            continue
        # 两端都属于锁定夹爪：相对位姿固定，排除
        if j1 in locked_gripper and j2 in locked_gripper:
            to_remove.append(idx)
            continue
        p1 = int(model.parents[j1]) if 0 <= j1 < njoints else -1
        p2 = int(model.parents[j2]) if 0 <= j2 < njoints else -1
        if p1 == j2 or p2 == j1:
            to_remove.append(idx)
    for idx in reversed(to_remove):
        geom_model.removeCollisionPair(geom_model.collisionPairs[idx])


def build_pose_validator(
    urdf_path: Optional[str] = None,
    limit_margin: float = 0.05,
    enable_collision: bool = True,
    enable_ground: bool = True,
    ground_z: float = 0.0,
    ground_exclude: Tuple[str, ...] = ("base_link",),
) -> PoseValidator:
    if urdf_path is None:
        urdf_path = str(default_urdf_path())

    package_dir = str(Path(urdf_path).resolve().parent.parent)
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()

    geom_model = None
    geom_data = None
    collision_enabled = False

    if enable_collision:
        try:
            geom_model = pin.buildGeomFromUrdf(
                model,
                urdf_path,
                pin.GeometryType.COLLISION,
                package_dirs=[package_dir],
            )
            geom_model.addAllCollisionPairs()
            _remove_adjacent_collision_pairs(model, geom_model)
            if enable_ground:
                gid = _add_ground_plane(geom_model, ground_z, ground_exclude)
                if gid is None:
                    print("[pose_validator] 警告: hppfcl 不可用，未加入水平面检查")
            geom_data = pin.GeometryData(geom_model)
            collision_enabled = True
        except Exception as exc:
            print(f"[pose_validator] 碰撞模型加载失败，降级为仅限位检查: {exc}")

    return PoseValidator(
        model=model,
        data=data,
        collision_enabled=collision_enabled,
        geom_model=geom_model,
        geom_data=geom_data,
        limit_margin=limit_margin,
    )
