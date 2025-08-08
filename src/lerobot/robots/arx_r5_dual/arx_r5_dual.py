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

import logging
import time
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .arx_r5_interface import ARXR5Interface
from .config_arx_r5_dual import ARXR5DualConfig

logger = logging.getLogger(__name__)

GRIPPER_POS_OPEN = 4.8
GRIPPER_POS_CLOSE = 0.0


class ARXR5Dual(Robot):
    """
    ARX R5 dual arm robot implementation for LeRobot.

    This robot consists of two ARX R5 robot arms working in coordination,
    with optional camera systems for visual feedback.
    """

    config_class = ARXR5DualConfig
    name = "arx_r5_dual"

    def __init__(self, config: ARXR5DualConfig):
        super().__init__(config)
        self.config = config

        # Initialize robot arms
        self.left_arm: ARXR5Interface | None = None
        self.right_arm: ARXR5Interface | None = None

        # Initialize cameras
        self.cameras = None

        # Control parameters
        self.control_dt = config.control_dt

        # State tracking
        self._is_connected = False
        self._is_calibrated = True  # ARX R5 doesn't require explicit calibration

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Get motor features for both arms with individual joint features."""
        features = {}

        # Left arm joints
        for i in range(1, 7):  # joint_1 to joint_6
            features[f"left_joint_{i}.pos"] = float
        features["left_gripper.pos"] = float

        # Right arm joints
        for i in range(1, 7):  # joint_1 to joint_6
            features[f"right_joint_{i}.pos"] = float
        features["right_gripper.pos"] = float

        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Get camera features."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in (self.config.cameras or {})
        }

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        """
        Define the structure of observations from the robot.

        Returns:
            Dict describing observation structure with individual joint features:
            - Individual joint positions for each arm (left_joint_1.pos, etc.)
            - Individual gripper positions (left_gripper.pos, right_gripper.pos)
            - Camera images (if configured)
        """
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict[str, type]:
        """
        Define the structure of actions expected by the robot.

        Returns:
            Dict describing action structure with individual joint features:
            - Individual joint positions for each arm (left_joint_1.pos, etc.)
            - Individual gripper positions (left_gripper.pos, right_gripper.pos)
        """
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to both robot arms and cameras.

        Args:
            calibrate: Whether to calibrate after connecting (not used for ARX R5)
        """
        if self._is_connected:
            raise DeviceAlreadyConnectedError("ARX R5 dual robot is already connected")

        try:
            # Initialize left arm
            logger.info(f"Connecting to left arm on {self.config.left_arm_can_port}")
            self.left_arm = ARXR5Interface(can_port=self.config.left_arm_can_port, dt=self.control_dt)

            # Initialize right arm
            logger.info(f"Connecting to right arm on {self.config.right_arm_can_port}")
            self.right_arm = ARXR5Interface(can_port=self.config.right_arm_can_port, dt=self.control_dt)

            # Initialize cameras if configured
            if self.config.cameras:
                logger.info("Initializing cameras")
                self.cameras = make_cameras_from_configs(self.config.cameras)
                for camera_name, camera in self.cameras.items():
                    camera.connect()
                    logger.info(f"Connected to camera: {camera_name}")

            self._is_connected = True
            logger.info("ARX R5 dual robot connected successfully")

            # Move to home position if calibrate is requested
            if calibrate:
                self.calibrate()

        except Exception as e:
            logger.error(f"Failed to connect ARX R5 dual robot: {e}")
            self.disconnect()  # Clean up partial connections
            raise

    @property
    def is_calibrated(self) -> bool:
        """Check if robot is calibrated."""
        return self._is_calibrated and self._is_connected

    def calibrate(self) -> None:
        """
        Calibrate the robot by moving both arms to home position.
        """
        if not self._is_connected:
            raise DeviceNotConnectedError("Robot not connected")

        logger.info("Calibrating ARX R5 dual robot - moving to home position")

        try:
            # Move both arms to home position
            if self.left_arm:
                self.left_arm.go_home()
            if self.right_arm:
                self.right_arm.go_home()

            # Wait for movement to complete
            time.sleep(3.0)

            self._is_calibrated = True
            logger.info("ARX R5 dual robot calibration completed")

        except Exception as e:
            logger.error(f"Failed to calibrate ARX R5 dual robot: {e}")
            self._is_calibrated = False
            raise

    def configure(self) -> None:
        """
        Configure the robot arms for operation.
        """
        if not self._is_connected:
            raise DeviceNotConnectedError("Robot not connected")

        logger.info("Configuring ARX R5 dual robot")

        try:
            # Enable gravity compensation on both arms
            if self.left_arm:
                self.left_arm.gravity_compensation()
            if self.right_arm:
                self.right_arm.gravity_compensation()

            logger.info("ARX R5 dual robot configuration completed")

        except Exception as e:
            logger.error(f"Failed to configure ARX R5 dual robot: {e}")
            raise

    def get_observation(self) -> dict[str, Any]:
        """
        Get current observation from the robot.

        Returns:
            Dict containing individual joint features:
            - left_joint_1.pos, left_joint_2.pos, ..., left_joint_6.pos
            - right_joint_1.pos, right_joint_2.pos, ..., right_joint_6.pos
            - left_gripper.pos, right_gripper.pos
            - Camera images (if configured)
        """
        if not self._is_connected:
            raise DeviceNotConnectedError("Robot not connected")

        observation = {}

        try:
            # Get joint states from both arms
            left_joint_pos = self.left_arm.get_joint_positions() if self.left_arm else [0.0] * 7
            right_joint_pos = self.right_arm.get_joint_positions() if self.right_arm else [0.0] * 7

            # Get gripper positions
            left_gripper_pos = left_joint_pos[6]
            left_gripper_pos = 1 - left_gripper_pos / (GRIPPER_POS_OPEN - GRIPPER_POS_CLOSE)
            right_gripper_pos = right_joint_pos[6]
            right_gripper_pos = 1 - right_gripper_pos / (GRIPPER_POS_OPEN - GRIPPER_POS_CLOSE)

            # Add individual joint positions
            for i in range(6):
                observation[f"left_joint_{i + 1}.pos"] = np.float32(left_joint_pos[i])
                observation[f"right_joint_{i + 1}.pos"] = np.float32(right_joint_pos[i])

            # Add gripper positions
            observation["left_gripper.pos"] = np.float32(left_gripper_pos)
            observation["right_gripper.pos"] = np.float32(right_gripper_pos)

            # Get camera images if available
            if self.cameras:
                for camera_name, camera in self.cameras.items():
                    image = camera.read()
                    observation[camera_name] = image

            observation_to_log = {k: v for k, v in observation.items() if not isinstance(v, np.ndarray)}
            logger.info(f"[get_observation] observation.state: {observation_to_log}")
            return observation

        except Exception as e:
            logger.error(f"Failed to get observation: {e}")
            raise

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Send action commands to both robot arms.

        Args:
            action: Dict containing individual joint features:
                - left_joint_1.pos, left_joint_2.pos, ..., left_joint_6.pos
                - right_joint_1.pos, right_joint_2.pos, ..., right_joint_6.pos
                - left_gripper.pos, right_gripper.pos

        Returns:
            Dict containing the actual action sent (potentially clipped)
        """
        if not self._is_connected:
            raise DeviceNotConnectedError("Robot not connected")

        logger.info(f"[send_action] action: {action}")

        try:
            # Extract individual joint targets
            left_joint_targets = []
            right_joint_targets = []

            for i in range(1, 7):  # joint_1 to joint_6
                left_joint_targets.append(float(action[f"left_joint_{i}.pos"]))
                right_joint_targets.append(float(action[f"right_joint_{i}.pos"]))

            left_gripper_target = float(action["left_gripper.pos"])
            left_gripper_target = (
                left_gripper_target * (GRIPPER_POS_CLOSE - GRIPPER_POS_OPEN) + GRIPPER_POS_OPEN
            )
            right_gripper_target = float(action["right_gripper.pos"])
            right_gripper_target = (
                right_gripper_target * (GRIPPER_POS_CLOSE - GRIPPER_POS_OPEN) + GRIPPER_POS_OPEN
            )

            print(f"Left gripper target: {left_gripper_target}, Right gripper target: {right_gripper_target}")

            # Convert to numpy arrays for clipping
            left_joint_targets = np.array(left_joint_targets)
            right_joint_targets = np.array(right_joint_targets)

            # Apply joint limits (basic clipping)
            left_joint_targets = np.clip(
                left_joint_targets, self.config.joint_limits_lower[:6], self.config.joint_limits_upper[:6]
            )
            right_joint_targets = np.clip(
                right_joint_targets,
                self.config.joint_limits_lower[6:12],
                self.config.joint_limits_upper[6:12],
            )

            # Apply gripper limits
            left_gripper_target = np.clip(
                left_gripper_target, self.config.gripper_limits_lower[0], self.config.gripper_limits_upper[0]
            )
            right_gripper_target = np.clip(
                right_gripper_target, self.config.gripper_limits_lower[1], self.config.gripper_limits_upper[1]
            )

            # Send commands to arms
            if self.left_arm:
                self.left_arm.set_joint_positions(left_joint_targets.tolist())
                self.left_arm.set_catch_pos(float(left_gripper_target))

            if self.right_arm:
                self.right_arm.set_joint_positions(right_joint_targets.tolist())
                self.right_arm.set_catch_pos(float(right_gripper_target))

            # Return the actual action sent (after clipping) in individual joint format
            actual_action = {}
            for i in range(6):
                actual_action[f"left_joint_{i + 1}.pos"] = np.float32(left_joint_targets[i])
                actual_action[f"right_joint_{i + 1}.pos"] = np.float32(right_joint_targets[i])

            actual_action["left_gripper.pos"] = np.float32(left_gripper_target)
            actual_action["right_gripper.pos"] = np.float32(right_gripper_target)

            return actual_action

        except Exception as e:
            logger.error(f"Failed to send action: {e}")
            raise

    def disconnect(self) -> None:
        """Disconnect from robot arms and cameras."""
        logger.info("Disconnecting ARX R5 dual robot")

        try:
            # Disconnect arms
            if self.left_arm:
                self.left_arm.disconnect()
                self.left_arm = None

            if self.right_arm:
                self.right_arm.disconnect()
                self.right_arm = None

            # Disconnect cameras
            if self.cameras:
                for camera_name, camera in self.cameras.items():
                    camera.disconnect()
                    logger.info(f"Disconnected camera: {camera_name}")
                self.cameras = None

            self._is_connected = False
            self._is_calibrated = False
            logger.info("ARX R5 dual robot disconnected")

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
            # Still mark as disconnected even if cleanup failed
            self._is_connected = False
            self._is_calibrated = False
