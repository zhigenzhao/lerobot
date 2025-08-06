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

"""
ARX R5 robot arm interface for LeRobot.

This module provides an interface for the ARX R5 robot arm based on the
original implementation.
"""

import logging
import os
from typing import List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import arx_r5_python.arx_r5_python as arx
    import meshcat.transformations as tf

    ARX_AVAILABLE = True
except ImportError:
    logger.warning("ARX R5 SDK not available. ARX R5 robot will not work.")
    ARX_AVAILABLE = False


class ARXR5Interface:
    """
    Interface for a single ARX R5 robot arm.

    Args:
        can_port (str): CAN port identifier (e.g., "can0", "can1")
        dt (float): Control timestep in seconds
    """

    def __init__(
        self,
        can_port: str = "can0",
        dt: float = 0.01,
    ):
        if not ARX_AVAILABLE:
            raise RuntimeError(
                "ARX R5 SDK not available. Please install the required dependencies:\n"
                "- arx Python package\n"
                "- transformations package"
            )

        self.dt = dt
        self.can_port = can_port
        self._is_connected = False

        # Initialize the ARX interface
        file_path = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(file_path, "assets/R5a/R5a.urdf")

        # Check if URDF file exists, if not use a default path
        if not os.path.exists(urdf_path):
            logger.warning(f"URDF file not found at {urdf_path}. Using default ARX R5 URDF.")
            # You may need to adjust this path based on your ARX SDK installation
            urdf_path = "R5a.urdf"

        try:
            self.arm = arx.InterfacesPy(urdf_path, can_port, 0)
            self.arm.arx_x(500, 2000, 10)
            self._is_connected = True
            logger.info(f"ARX R5 interface initialized on {can_port}")
        except Exception as e:
            logger.error(f"Failed to initialize ARX R5 interface: {e}")
            self._is_connected = False
            raise

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self._is_connected

    def disconnect(self):
        """Disconnect from the robot."""
        self._is_connected = False
        logger.info(f"Disconnected from ARX R5 on {self.can_port}")

    def get_joint_names(self) -> List[str]:
        """
        Get the names of all joints in the arm.

        Returns:
            List[str]: List of joint names. Shape: (num_joints,)
        """
        # Default joint names for ARX R5 - adjust if needed
        return [f"joint_{i+1}" for i in range(6)]

    def go_home(self) -> bool:
        """
        Move the robot arm to a pre-defined home pose.

        Returns:
            bool: True if the action was successful, False otherwise
        """
        if not self._is_connected:
            raise RuntimeError("Robot not connected")

        try:
            self.arm.set_arm_status(1)
            return True
        except Exception as e:
            logger.error(f"Failed to move to home position: {e}")
            return False

    def gravity_compensation(self) -> bool:
        """Enable gravity compensation mode."""
        if not self._is_connected:
            raise RuntimeError("Robot not connected")

        try:
            self.arm.set_arm_status(3)
            return True
        except Exception as e:
            logger.error(f"Failed to enable gravity compensation: {e}")
            return False

    def protect_mode(self) -> bool:
        """Enable protect mode."""
        if not self._is_connected:
            raise RuntimeError("Robot not connected")

        try:
            self.arm.set_arm_status(2)
            return True
        except Exception as e:
            logger.error(f"Failed to enable protect mode: {e}")
            return False

    def set_joint_positions(
        self,
        positions: Union[float, List[float], np.ndarray],
        **kwargs,
    ) -> bool:
        """
        Move the arm to the given joint position(s).

        Args:
            positions: Desired joint position(s). Shape: (6)
            **kwargs: Additional arguments
        """
        if not self._is_connected:
            raise RuntimeError("Robot not connected")

        try:
            self.arm.set_joint_positions(positions)
            self.arm.set_arm_status(5)
            return True
        except Exception as e:
            logger.error(f"Failed to set joint positions: {e}")
            return False

    def set_ee_pose(
        self,
        pos: Optional[Union[List[float], np.ndarray]] = None,
        quat: Optional[Union[List[float], np.ndarray]] = None,
        **kwargs,
    ) -> bool:
        """
        Move the end effector to the given pose.

        Args:
            pos: Desired position [x, y, z]. Shape: (3,)
            quat: Desired orientation (quaternion [w, x, y, z]). Shape: (4,)
            **kwargs: Additional arguments
        """
        if not self._is_connected:
            raise RuntimeError("Robot not connected")

        if pos is None or quat is None:
            raise ValueError("Both position and quaternion must be provided")

        try:
            pose = [pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]
            self.arm.set_ee_pose(pose)
            self.arm.set_arm_status(4)
            return True
        except Exception as e:
            logger.error(f"Failed to set end effector pose: {e}")
            return False

    def set_ee_pose_xyzrpy(
        self,
        xyzrpy: Optional[Union[List[float], np.ndarray]] = None,
        **kwargs,
    ) -> bool:
        """
        Move the end effector to the given pose.

        Args:
            xyzrpy: Desired position [x, y, z, roll, pitch, yaw]. Shape: (6,)
            **kwargs: Additional arguments
        """
        if not self._is_connected:
            raise RuntimeError("Robot not connected")

        if xyzrpy is None:
            raise ValueError("xyzrpy must be provided")

        try:
            quat = tf.quaternion_from_euler(xyzrpy[3], xyzrpy[4], xyzrpy[5])
            pose = [xyzrpy[0], xyzrpy[1], xyzrpy[2], quat[0], quat[1], quat[2], quat[3]]
            self.arm.set_ee_pose(pose)
            self.arm.set_arm_status(4)
            return True
        except Exception as e:
            logger.error(f"Failed to set end effector pose (xyzrpy): {e}")
            return False

    def set_catch_pos(self, pos: float) -> bool:
        """
        Set gripper position.

        Args:
            pos: Gripper position
        """
        if not self._is_connected:
            raise RuntimeError("Robot not connected")

        try:
            self.arm.set_catch(pos)
            self.arm.set_arm_status(5)
            return True
        except Exception as e:
            logger.error(f"Failed to set gripper position: {e}")
            return False

    def get_joint_positions(
        self, joint_names: Optional[Union[str, List[str]]] = None
    ) -> Union[float, List[float]]:
        """
        Get the current joint position(s) of the arm.

        Args:
            joint_names: Name(s) of the joint(s) to get positions for.
                        If None, return positions for all joints.

        Returns:
            Current joint positions
        """
        if not self._is_connected:
            raise RuntimeError("Robot not connected")

        try:
            return self.arm.get_joint_positions()
        except Exception as e:
            logger.error(f"Failed to get joint positions: {e}")
            return [0.0] * 6

    def get_joint_velocities(
        self, joint_names: Optional[Union[str, List[str]]] = None
    ) -> Union[float, List[float]]:
        """
        Get the current joint velocity(ies) of the arm.

        Args:
            joint_names: Name(s) of the joint(s) to get velocities for.
                        If None, return velocities for all joints.

        Returns:
            Current joint velocities
        """
        if not self._is_connected:
            raise RuntimeError("Robot not connected")

        try:
            return self.arm.get_joint_velocities()
        except Exception as e:
            logger.error(f"Failed to get joint velocities: {e}")
            return [0.0] * 6

    def get_joint_currents(
        self, joint_names: Optional[Union[str, List[str]]] = None
    ) -> Union[float, List[float]]:
        """
        Get the current joint currents of the arm.

        Args:
            joint_names: Name(s) of the joint(s) to get currents for.
                        If None, return currents for all joints.

        Returns:
            Current joint currents
        """
        if not self._is_connected:
            raise RuntimeError("Robot not connected")

        try:
            return self.arm.get_joint_currents()
        except Exception as e:
            logger.error(f"Failed to get joint currents: {e}")
            return [0.0] * 6

    def get_ee_pose(self) -> List[float]:
        """
        Get the current end effector pose of the arm.

        Returns:
            End effector pose as [x, y, z, qw, qx, qy, qz]
        """
        if not self._is_connected:
            raise RuntimeError("Robot not connected")

        try:
            return self.arm.get_ee_pose()
        except Exception as e:
            logger.error(f"Failed to get end effector pose: {e}")
            return [0.0] * 7

    def get_ee_pose_xyzrpy(self) -> np.ndarray:
        """
        Get the current end effector pose as xyz + rpy.

        Returns:
            End effector pose as [x, y, z, roll, pitch, yaw]
        """
        if not self._is_connected:
            raise RuntimeError("Robot not connected")

        try:
            xyzwxyz = self.arm.get_ee_pose()
            quat_array = np.array([xyzwxyz[3], xyzwxyz[4], xyzwxyz[5], xyzwxyz[6]])
            roll, pitch, yaw = tf.euler_from_quaternion(quat_array)
            xyzrpy = np.array([xyzwxyz[0], xyzwxyz[1], xyzwxyz[2], roll, pitch, yaw])
            return xyzrpy
        except Exception as e:
            logger.error(f"Failed to get end effector pose (xyzrpy): {e}")
            return np.array([0.0] * 6)

    def __del__(self):
        """Destructor to clean up resources."""
        if hasattr(self, "_is_connected") and self._is_connected:
            self.disconnect()
        logger.debug("ARXR5Interface is being deleted")
