# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("arx_r5_dual")
@dataclass(kw_only=True)
class ARXR5DualConfig(RobotConfig):
    """Configuration for ARX R5 dual arm robot."""

    # CAN bus configuration for the two arms
    left_arm_can_port: str = "can1"
    right_arm_can_port: str = "can3"

    # Control parameters
    control_dt: float = 0.02  # Control timestep in seconds

    # Home position for both arms (6 joints each)
    left_arm_home_position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    right_arm_home_position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Gripper home positions
    left_gripper_home_position: float = 0.0
    right_gripper_home_position: float = 0.0

    # Joint limits (optional, can be loaded from URDF)
    joint_limits_lower: list[float] = field(default_factory=lambda: [-3.14] * 12)  # 6 joints per arm
    joint_limits_upper: list[float] = field(default_factory=lambda: [3.14] * 12)

    # Gripper limits
    gripper_limits_lower: list[float] = field(default_factory=lambda: [0.0, 0.0])
    gripper_limits_upper: list[float] = field(default_factory=lambda: [1.0, 1.0])

    # Camera configuration with default RealSense cameras
    # Note: Using keys that will map to policy-expected names after hw_to_dataset_features transformation
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "base_image": RealSenseCameraConfig(
                serial_number_or_name="215222077461", fps=60, width=424, height=240, use_depth=False
            ),
            "left_wrist_image": RealSenseCameraConfig(
                serial_number_or_name="218622272499", fps=60, width=424, height=240, use_depth=False
            ),
            "right_wrist_image": RealSenseCameraConfig(
                serial_number_or_name="218622272014", fps=60, width=424, height=240, use_depth=False
            ),
        }
    )

    def __post_init__(self):
        super().__post_init__()

        # Validate home positions
        if len(self.left_arm_home_position) != 6:
            raise ValueError("left_arm_home_position must have 6 elements")
        if len(self.right_arm_home_position) != 6:
            raise ValueError("right_arm_home_position must have 6 elements")

        # Validate joint limits
        if len(self.joint_limits_lower) != 12:
            raise ValueError("joint_limits_lower must have 12 elements (6 per arm)")
        if len(self.joint_limits_upper) != 12:
            raise ValueError("joint_limits_upper must have 12 elements (6 per arm)")
