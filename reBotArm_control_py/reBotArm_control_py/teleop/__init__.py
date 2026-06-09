"""teleop 遥操作模块 — 手柄（Xbox / 通用 HID）增量式笛卡尔遥操作。

使用示例::

    from reBotArm_control_py.actuator import RobotArm, Gripper
    from reBotArm_control_py.teleop import (
        LinuxJoystick, XboxTeleopController, detect_controller,
    )

    detection = detect_controller("/dev/input/js0")
    joy = LinuxJoystick("/dev/input/js0")
    joy.connect()

    arm = RobotArm()
    arm.connect(); arm.mode_mit(); arm.enable()

    teleop = XboxTeleopController(arm, joy, detection.profile)
    teleop.start()        # 启动 100Hz 控制循环（后台线程）
    ...
    teleop.stop()
"""

from .joystick import LinuxJoystick
from .controller_profiles import (
    AxisBinding,
    TriggerBinding,
    StickMap,
    ButtonMap,
    ControllerProfile,
    ControllerDetection,
    PROFILES,
    list_profiles,
    get_profile,
    detect_controller,
    read_controller_metadata,
)
from .xbox_controller import (
    XboxTeleopController,
    SPEED_LEVELS,
    HOME_POSITIONS,
    ZERO_POSITIONS,
)

__all__ = [
    # joystick
    "LinuxJoystick",
    # profiles
    "AxisBinding",
    "TriggerBinding",
    "StickMap",
    "ButtonMap",
    "ControllerProfile",
    "ControllerDetection",
    "PROFILES",
    "list_profiles",
    "get_profile",
    "detect_controller",
    "read_controller_metadata",
    # controller
    "XboxTeleopController",
    "SPEED_LEVELS",
    "HOME_POSITIONS",
    "ZERO_POSITIONS",
]
