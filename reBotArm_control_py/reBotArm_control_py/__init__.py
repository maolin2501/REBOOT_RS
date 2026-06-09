"""reBotArm_control_py - reBotArm 机械臂 Python 控制库。"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["actuator", "kinematics", "dynamics", "calibration", "controllers", "trajectory", "teleop"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
