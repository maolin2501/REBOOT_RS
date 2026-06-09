"""手柄逻辑映射 profile（逻辑功能 → 物理轴/按钮索引）。

用数据类把"逻辑功能"绑定到手柄的"物理索引"，并支持按
``VID:PID → 设备名正则 → fallback`` 三级优先级自动识别手柄型号。
此模块与机械臂无关，移植自 EL-A3 SDK。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class AxisBinding:
    """摇杆/方向键轴绑定，支持反向。"""
    index: Optional[int]
    invert: bool = False

    def read(self, axes) -> float:
        if self.index is None or self.index >= len(axes):
            return 0.0
        value = axes[self.index]
        return -value if self.invert else value


@dataclass(frozen=True)
class TriggerBinding:
    """扳机绑定：既可走模拟轴（线性重映射到 [0,1]），也可走数字按钮。"""
    index: Optional[int] = None
    button: Optional[int] = None
    scale: float = 0.5
    offset: float = 0.5
    clamp_min: float = 0.0
    clamp_max: float = 1.0

    def read(self, axes, buttons) -> float:
        if self.button is not None:
            if self.button >= len(buttons):
                return 0.0
            return 1.0 if buttons[self.button] else 0.0

        if self.index is None or self.index >= len(axes):
            return 0.0

        value = axes[self.index] * self.scale + self.offset
        if value < self.clamp_min:
            return self.clamp_min
        if value > self.clamp_max:
            return self.clamp_max
        return value


@dataclass(frozen=True)
class StickMap:
    lx: AxisBinding
    ly: AxisBinding
    rx: AxisBinding
    ry: AxisBinding
    dpad_x: AxisBinding
    dpad_y: AxisBinding
    lt: TriggerBinding
    rt: TriggerBinding


@dataclass(frozen=True)
class ButtonMap:
    south: Optional[int]   # A
    east: Optional[int]    # B
    west: Optional[int]    # X
    north: Optional[int]   # Y
    lb: Optional[int]
    rb: Optional[int]
    back: Optional[int]
    start: Optional[int]


@dataclass(frozen=True)
class ControllerProfile:
    profile_id: str
    display_name: str
    description: str
    sticks: StickMap
    buttons: ButtonMap
    default_deadzone: float = 0.15
    match_vid_pid: Tuple[Tuple[str, str], ...] = ()
    match_name_patterns: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ControllerDetection:
    device: str
    resolved_device: str
    name: str
    vendor: str
    product: str
    profile: ControllerProfile
    source: str


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _js_sysfs_dir(device: str) -> Optional[str]:
    resolved = os.path.realpath(device)
    name = os.path.basename(resolved)
    if not name.startswith("js"):
        return None
    sysfs_dir = os.path.join("/sys/class/input", name, "device")
    return sysfs_dir if os.path.isdir(sysfs_dir) else None


def read_controller_metadata(device: str) -> Dict[str, str]:
    """从 sysfs 读取手柄的名称与 VID/PID，用于自动识别 profile。"""
    resolved = os.path.realpath(device)
    sysfs_dir = _js_sysfs_dir(device)
    name = ""
    vendor = ""
    product = ""
    if sysfs_dir:
        name = _read_text(os.path.join(sysfs_dir, "name"))
        vendor = _read_text(os.path.join(sysfs_dir, "id", "vendor")).lower()
        product = _read_text(os.path.join(sysfs_dir, "id", "product")).lower()
    return {
        "device": device,
        "resolved_device": resolved,
        "name": name,
        "vendor": vendor,
        "product": product,
    }


PROFILES: Dict[str, ControllerProfile] = {
    "xbox_default": ControllerProfile(
        profile_id="xbox_default",
        display_name="Xbox (xpad/xinput)",
        description="标准 Xbox 布局（Linux xpad 风格轴序）。",
        sticks=StickMap(
            lx=AxisBinding(0),
            ly=AxisBinding(1),
            lt=TriggerBinding(index=2, scale=0.5, offset=0.5),
            rx=AxisBinding(3),
            ry=AxisBinding(4),
            rt=TriggerBinding(index=5, scale=0.5, offset=0.5),
            dpad_x=AxisBinding(6),
            dpad_y=AxisBinding(7),
        ),
        buttons=ButtonMap(
            south=0,
            east=1,
            west=2,
            north=3,
            lb=4,
            rb=5,
            back=6,
            start=7,
        ),
        match_name_patterns=("xbox", "x-input", "xinput", "microsoft"),
    ),
    "zikway_3537_1041": ControllerProfile(
        profile_id="zikway_3537_1041",
        display_name="Zikway HID gamepad",
        description="VID:PID 3537:1041 上检测到的通用 HID 布局。",
        sticks=StickMap(
            lx=AxisBinding(0),
            ly=AxisBinding(1),
            lt=TriggerBinding(index=4, scale=0.5, offset=0.5),
            rx=AxisBinding(2),
            ry=AxisBinding(3),
            rt=TriggerBinding(index=5, scale=0.5, offset=0.5),
            dpad_x=AxisBinding(6),
            dpad_y=AxisBinding(7),
        ),
        buttons=ButtonMap(
            south=0,
            east=1,
            west=3,
            north=4,
            lb=6,
            rb=7,
            back=10,
            start=11,
        ),
        match_vid_pid=(("3537", "1041"),),
        match_name_patterns=("zikway",),
    ),
    "generic_hid": ControllerProfile(
        profile_id="generic_hid",
        display_name="Generic HID gamepad",
        description="8 轴 / 16 键 HID 手柄的兜底布局。",
        sticks=StickMap(
            lx=AxisBinding(0),
            ly=AxisBinding(1),
            lt=TriggerBinding(index=4, scale=0.5, offset=0.5),
            rx=AxisBinding(2),
            ry=AxisBinding(3),
            rt=TriggerBinding(index=5, scale=0.5, offset=0.5),
            dpad_x=AxisBinding(6),
            dpad_y=AxisBinding(7),
        ),
        buttons=ButtonMap(
            south=0,
            east=1,
            west=3,
            north=4,
            lb=6,
            rb=7,
            back=10,
            start=11,
        ),
    ),
}


def list_profiles() -> Iterable[ControllerProfile]:
    return PROFILES.values()


def get_profile(profile_id: str) -> ControllerProfile:
    if profile_id not in PROFILES:
        raise KeyError(f"未知的手柄 profile: {profile_id}")
    return PROFILES[profile_id]


def detect_controller(device: str, requested_profile: str = "auto") -> ControllerDetection:
    """自动识别手柄并返回对应 profile。

    优先级：显式指定 → VID:PID 精确匹配 → 设备名正则匹配 → fallback(generic_hid)。
    """
    metadata = read_controller_metadata(device)

    if requested_profile != "auto":
        return ControllerDetection(
            device=device,
            resolved_device=metadata["resolved_device"],
            name=metadata["name"],
            vendor=metadata["vendor"],
            product=metadata["product"],
            profile=get_profile(requested_profile),
            source="explicit",
        )

    vendor = metadata["vendor"]
    product = metadata["product"]
    name = metadata["name"].lower()

    for profile in PROFILES.values():
        if (vendor, product) in profile.match_vid_pid:
            return ControllerDetection(
                device=device,
                resolved_device=metadata["resolved_device"],
                name=metadata["name"],
                vendor=vendor,
                product=product,
                profile=profile,
                source="vid_pid",
            )

    for profile in PROFILES.values():
        if any(re.search(pattern, name) for pattern in profile.match_name_patterns):
            return ControllerDetection(
                device=device,
                resolved_device=metadata["resolved_device"],
                name=metadata["name"],
                vendor=vendor,
                product=product,
                profile=profile,
                source="name",
            )

    return ControllerDetection(
        device=device,
        resolved_device=metadata["resolved_device"],
        name=metadata["name"],
        vendor=vendor,
        product=product,
        profile=get_profile("generic_hid"),
        source="fallback",
    )
