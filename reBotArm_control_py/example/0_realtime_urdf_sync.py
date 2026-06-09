#!/usr/bin/env python3
"""实时电机位置同步 URDF 模型显示 — 基于 can0 全0 MIT 指令获取反馈。

原理:
    向所有电机持续发送全 0 MIT 帧 (pos=vel=kp=kd=tau=0)，
    电机处于零阻抗、不出力状态（可自由拖动），
    但会在每帧返回真实的 pos/vel/torq 反馈。
    主线程以 ~50Hz 刷新率将反馈位置写入 MeshCat URDF 模型。

    全程不调用 enable，电机始终不使能，不会主动运动。

用法:
    python example/0_realtime_urdf_sync.py

前置:
    sudo ip link set can0 type can bitrate 1000000
    sudo ip link set can0 up
"""
import sys
import time
import signal
import threading
from pathlib import Path

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motorbridge import Controller, Mode, CallError

CHANNEL = "can0"

URDF_PATH = str(
    Path(__file__).resolve().parents[1]
    / "urdf" / "00-arm-rs_asm-v3" / "urdf" / "00-arm-rs_asm-v3.urdf"
)
MESH_DIR = str(
    Path(__file__).resolve().parents[1] / "urdf" / "00-arm-rs_asm-v3"
)

MOTORS = [
    {"name": "joint1", "id": 0x01, "host": 0xFD, "model": "rs-06"},
    {"name": "joint2", "id": 0x02, "host": 0xFD, "model": "rs-06"},
    {"name": "joint3", "id": 0x03, "host": 0xFD, "model": "rs-06"},
    {"name": "joint4", "id": 0x04, "host": 0xFD, "model": "rs-00"},
    {"name": "joint5", "id": 0x05, "host": 0xFD, "model": "rs-00"},
    {"name": "joint6", "id": 0x06, "host": 0xFD, "model": "rs-00"},
]

MIT_RATE_HZ = 200
VIZ_RATE_HZ = 50


# ──────────────────────────────────────────────────────────────────────────────
# 电机反馈
# ──────────────────────────────────────────────────────────────────────────────

class MotorFeedbackLoop:
    """后台线程：持续发送全 0 MIT 指令并收集反馈位置。"""

    def __init__(self, channel: str):
        self._ctrl = Controller(channel)
        self._motors = []
        for m in MOTORS:
            mot = self._ctrl.add_robstride_motor(m["id"], m["host"], m["model"])
            self._motors.append(mot)

        self._positions = np.zeros(len(MOTORS))
        self._velocities = np.zeros(len(MOTORS))
        self._torques = np.zeros(len(MOTORS))
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        try:
            self._ctrl.shutdown()
        except Exception:
            pass
        try:
            self._ctrl.close()
        except Exception:
            pass

    def get_positions(self) -> np.ndarray:
        with self._lock:
            return self._positions.copy()

    def get_state(self):
        with self._lock:
            return self._positions.copy(), self._velocities.copy(), self._torques.copy()

    def _loop(self):
        dt = 1.0 / MIT_RATE_HZ
        while self._running:
            t0 = time.perf_counter()

            for mot in self._motors:
                try:
                    mot.send_mit(0.0, 0.0, 0.0, 0.0, 0.0)
                except CallError:
                    pass

            for mot in self._motors:
                try:
                    mot.request_feedback()
                except CallError:
                    pass

            try:
                self._ctrl.poll_feedback_once()
            except CallError:
                pass

            pos = np.zeros(len(self._motors))
            vel = np.zeros(len(self._motors))
            torq = np.zeros(len(self._motors))
            for i, mot in enumerate(self._motors):
                try:
                    st = mot.get_state()
                    if st is not None:
                        pos[i] = st.pos
                        vel[i] = st.vel
                        torq[i] = st.torq
                except CallError:
                    pass

            with self._lock:
                self._positions = pos
                self._velocities = vel
                self._torques = torq

            elapsed = time.perf_counter() - t0
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)


# ──────────────────────────────────────────────────────────────────────────────
# URDF 可视化
# ──────────────────────────────────────────────────────────────────────────────

class URDFVisualizer:
    """基于 Pinocchio + MeshCat 的 URDF 可视化器。"""

    def __init__(self):
        self._model = pin.buildModelFromUrdf(URDF_PATH)
        self._data = self._model.createData()
        self._visual_model = pin.buildGeomFromUrdf(
            self._model, URDF_PATH, pin.GeometryType.VISUAL,
            package_dirs=[MESH_DIR],
        )
        self._visual_data = self._visual_model.createData()

        self._meshcat_viz = meshcat.Visualizer(zmq_url=None)
        self._viz = MeshcatVisualizer(
            self._model,
            collision_model=None,
            visual_model=self._visual_model,
            data=self._data,
            visual_data=self._visual_data,
        )
        self._viz.initViewer(self._meshcat_viz, loadModel=False)
        self._viz.loadViewerModel()

    @property
    def url(self) -> str:
        return self._meshcat_viz.url()

    @property
    def nq(self) -> int:
        return self._model.nq

    def update(self, q: np.ndarray):
        self._viz.display(np.asarray(q, dtype=np.float64))


# ──────────────────────────────────────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  实时 URDF 模型同步 (全 0 MIT 反馈模式)")
    print("=" * 60)
    print(f"  通道     : {CHANNEL}")
    print(f"  MIT 频率 : {MIT_RATE_HZ} Hz")
    print(f"  显示频率 : {VIZ_RATE_HZ} Hz")
    print(f"  电机数   : {len(MOTORS)}")
    print()

    print("[1/2] 连接电机总线...")
    try:
        fb = MotorFeedbackLoop(CHANNEL)
    except CallError as e:
        print(f"[致命] 无法打开 {CHANNEL}: {e}")
        print("请确认 can0 已 up:")
        print("  sudo ip link set can0 type can bitrate 1000000")
        print("  sudo ip link set can0 up")
        sys.exit(1)

    print("[2/2] 加载 URDF 模型 + MeshCat 可视化...")
    viz = URDFVisualizer()
    print(f"  MeshCat 地址: {viz.url}")
    print(f"  模型自由度  : nq = {viz.nq}")
    print()

    fb.start()
    print("[运行中] 全 0 MIT 指令发送中（电机不使能、不出力）...")
    print("         拖动电机可看到 URDF 模型实时跟随。")
    print("         Ctrl+C 退出。\n")

    shutdown = threading.Event()

    def on_signal(sig, frame):
        shutdown.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    viz_dt = 1.0 / VIZ_RATE_HZ
    try:
        while not shutdown.is_set():
            t0 = time.perf_counter()

            pos = fb.get_positions()
            q = np.zeros(viz.nq)
            n = min(len(pos), viz.nq)
            q[:n] = pos[:n]
            viz.update(q)

            pos_deg = np.degrees(pos)
            row = "  ".join(f"{d:+7.2f}" for d in pos_deg)
            print(f"\r  关节角(deg): {row}", end="", flush=True)

            elapsed = time.perf_counter() - t0
            sleep_t = viz_dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n\n[退出] 停止反馈循环...")
        fb.stop()
        print("[完成]")


if __name__ == "__main__":
    main()
