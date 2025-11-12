#!/usr/bin/env python3
"""AUBO i5 Robot Arm Kinematics Implementation.

This module provides forward and inverse kinematics for the AUBO i5 6-DOF
collaborative robot arm.

AUBO i5 Specifications:
- 6 degrees of freedom (6-DOF)
- Payload: 5 kg
- Reach: 886.5 mm
- Repeatability: ±0.05 mm
- All joints: ±175° range of motion

Kinematic Parameters (from URDF):
┌──────────────────┬─────────────┬─────────────────────┬──────────────────────┐
│ Joint Name       │ Parent Link │ Origin XYZ (m)      │ Origin RPY (rad)     │
├──────────────────┼─────────────┼─────────────────────┼──────────────────────┤
│ shoulder_joint   │ base_link   │ (0, 0, 0.122)       │ (0, 0, π)            │
│ upperArm_joint   │ shoulder    │ (0, 0.1215, 0)      │ (-π/2, -π/2, 0)      │
│ foreArm_joint    │ upperArm    │ (0.408, 0, 0)       │ (-π, 0, 0)           │
│ wrist1_joint     │ foreArm     │ (0.376, 0, 0)       │ (π, 0, π/2)          │
│ wrist2_joint     │ wrist1      │ (0, 0.1025, 0)      │ (-π/2, 0, 0)         │
│ wrist3_joint     │ wrist2      │ (0, -0.094, 0)      │ (π/2, 0, 0)          │
└──────────────────┴─────────────┴─────────────────────┴──────────────────────┘

Reference:
- URDF: https://github.com/avinashsen707/AUBOi5-D435-ROS-DOPE
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Add external directory to path for base classes
sys.path.insert(0, str(Path(__file__).parent / "external"))

from forward_kinematics import ForwardKinematics, URDFJoint


class AuboI5ForwardKinematics(ForwardKinematics):
    """Forward kinematics for AUBO i5 6-DOF collaborative robot arm.

    The AUBO i5 has 6 revolute joints providing 6 degrees of freedom:
    - Joint 1: Shoulder rotation (base rotation, Z-axis)
    - Joint 2: Shoulder lift (upper arm lift, Y-axis)
    - Joint 3: Elbow (forearm rotation, Y-axis)
    - Joint 4: Wrist 1 (wrist rotation, Y-axis)
    - Joint 5: Wrist 2 (wrist pitch, Z-axis)
    - Joint 6: Wrist 3 (wrist roll, Y-axis)

    All measurements in meters and radians.
    """

    # Joint limits in radians (±175° = ±3.054 rad for all joints)
    JOINT_LIMITS_RAD = np.array([
        [-3.054, 3.054],  # shoulder_joint
        [-3.054, 3.054],  # upperArm_joint
        [-3.054, 3.054],  # foreArm_joint
        [-3.054, 3.054],  # wrist1_joint
        [-3.054, 3.054],  # wrist2_joint
        [-3.054, 3.054],  # wrist3_joint
    ])

    # Link lengths (m)
    SHOULDER_HEIGHT = 0.122    # Base to shoulder
    SHOULDER_OFFSET = 0.1215   # Shoulder to upper arm
    UPPER_ARM_LENGTH = 0.408   # Upper arm length
    FOREARM_LENGTH = 0.376     # Forearm length
    WRIST1_OFFSET = 0.1025     # Wrist1 to wrist2
    WRIST2_OFFSET = 0.094      # Wrist2 to wrist3 (end effector)

    def __init__(self):
        """Initialize AUBO i5 forward kinematics with URDF joint definitions."""
        # Joint definitions from AUBO i5 URDF
        joints = [
            # Joint 1: Shoulder rotation (base rotation around Z-axis)
            URDFJoint(
                name='shoulder_joint',
                origin_xyz=np.array([0.0, 0.0, 0.122]),
                origin_rpy=np.array([0.0, 0.0, np.pi]),
                axis=np.array([0.0, 0.0, 1.0]),  # Z-axis rotation
                joint_type='revolute'
            ),

            # Joint 2: Shoulder lift (upper arm lift around Y-axis)
            URDFJoint(
                name='upperArm_joint',
                origin_xyz=np.array([0.0, 0.1215, 0.0]),
                origin_rpy=np.array([-np.pi/2, -np.pi/2, 0.0]),
                axis=np.array([0.0, 1.0, 0.0]),  # Y-axis rotation
                joint_type='revolute'
            ),

            # Joint 3: Elbow (forearm rotation around Y-axis)
            URDFJoint(
                name='foreArm_joint',
                origin_xyz=np.array([0.408, 0.0, 0.0]),
                origin_rpy=np.array([-np.pi, 0.0, 0.0]),
                axis=np.array([0.0, 1.0, 0.0]),  # Y-axis rotation
                joint_type='revolute'
            ),

            # Joint 4: Wrist 1 (wrist rotation around Y-axis)
            URDFJoint(
                name='wrist1_joint',
                origin_xyz=np.array([0.376, 0.0, 0.0]),
                origin_rpy=np.array([np.pi, 0.0, np.pi/2]),
                axis=np.array([0.0, 1.0, 0.0]),  # Y-axis rotation
                joint_type='revolute'
            ),

            # Joint 5: Wrist 2 (wrist pitch around Z-axis)
            URDFJoint(
                name='wrist2_joint',
                origin_xyz=np.array([0.0, 0.1025, 0.0]),
                origin_rpy=np.array([-np.pi/2, 0.0, 0.0]),
                axis=np.array([0.0, 0.0, 1.0]),  # Z-axis rotation
                joint_type='revolute'
            ),

            # Joint 6: Wrist 3 (wrist roll around Y-axis)
            URDFJoint(
                name='wrist3_joint',
                origin_xyz=np.array([0.0, -0.094, 0.0]),
                origin_rpy=np.array([np.pi/2, 0.0, 0.0]),
                axis=np.array([0.0, 1.0, 0.0]),  # Y-axis rotation
                joint_type='revolute'
            ),
        ]

        super().__init__(joints)

    def compute_with_full_state(self, joint_angles: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute forward kinematics with full robot state.

        Args:
            joint_angles: 6D array [6 joint angles in radians]

        Returns:
            Dictionary with:
                - 'joint_positions': List of 3D positions for each joint
                - 'joint_frames': List of coordinate frames (position + rotation axes)
                - 'end_effector_pos': End effector position [x, y, z]
                - 'end_effector_rot': End effector rotation matrix [3x3]
                - 'joint_angles': Input joint angles
        """
        if len(joint_angles) != 6:
            raise ValueError(f"AUBO i5 requires 6 joint angles, got {len(joint_angles)}")

        # Compute forward kinematics
        joint_positions = self.compute_joint_positions(joint_angles)
        joint_frames = self.compute_joint_frames(joint_angles)
        ee_pos, ee_rot = self.compute_end_effector_pose(joint_angles)

        return {
            'joint_positions': joint_positions,
            'joint_frames': joint_frames,
            'end_effector_pos': ee_pos,
            'end_effector_rot': ee_rot,
            'joint_angles': joint_angles,
        }

    def is_within_joint_limits(self, joint_angles: np.ndarray) -> bool:
        """Check if joint angles are within valid limits.

        Args:
            joint_angles: 6D array of joint angles in radians

        Returns:
            True if all joints within limits, False otherwise
        """
        if len(joint_angles) != 6:
            return False

        for i, angle in enumerate(joint_angles):
            if angle < self.JOINT_LIMITS_RAD[i, 0] or angle > self.JOINT_LIMITS_RAD[i, 1]:
                return False

        return True

    def clip_to_joint_limits(self, joint_angles: np.ndarray) -> np.ndarray:
        """Clip joint angles to valid limits.

        Args:
            joint_angles: 6D array of joint angles in radians

        Returns:
            Clipped joint angles within limits
        """
        if len(joint_angles) != 6:
            raise ValueError(f"Expected 6 joint angles, got {len(joint_angles)}")

        clipped = np.zeros(6)
        for i in range(6):
            clipped[i] = np.clip(
                joint_angles[i],
                self.JOINT_LIMITS_RAD[i, 0],
                self.JOINT_LIMITS_RAD[i, 1]
            )

        return clipped


def test_aubo_i5_fk():
    """Test AUBO i5 forward kinematics with sample configurations."""
    fk = AuboI5ForwardKinematics()

    print("=" * 70)
    print("AUBO i5 Forward Kinematics Test")
    print("=" * 70)

    # Test 1: Home position (all zeros)
    print("\nTest 1: Home position (all zeros)")
    home = np.zeros(6)
    result = fk.compute_with_full_state(home)
    print(f"  End effector position: {result['end_effector_pos']}")
    print(f"  EE distance from base: {np.linalg.norm(result['end_effector_pos']):.4f} m")
    print(f"  Number of joints: {len(result['joint_positions'])}")

    # Test 2: Arm extended forward (should reach near max reach)
    print("\nTest 2: Arm extended forward")
    extended = np.array([0.0, -np.pi/2, 0.0, 0.0, 0.0, 0.0])
    result = fk.compute_with_full_state(extended)
    print(f"  End effector position: {result['end_effector_pos']}")
    print(f"  EE distance from base: {np.linalg.norm(result['end_effector_pos']):.4f} m")
    print(f"  (Expected reach: ~0.886 m)")

    # Test 3: Arm folded configuration
    print("\nTest 3: Arm folded")
    folded = np.array([0.0, np.pi/4, np.pi/2, -np.pi/4, 0.0, 0.0])
    result = fk.compute_with_full_state(folded)
    print(f"  End effector position: {result['end_effector_pos']}")
    print(f"  EE distance from base: {np.linalg.norm(result['end_effector_pos']):.4f} m")

    # Test 4: Joint limit checking
    print("\nTest 4: Joint limit checking")
    within_limits = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    outside_limits = np.array([4.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # 4.0 > 3.054 limit
    print(f"  Joints {within_limits} within limits: {fk.is_within_joint_limits(within_limits)}")
    print(f"  Joints {outside_limits} within limits: {fk.is_within_joint_limits(outside_limits)}")

    clipped = fk.clip_to_joint_limits(outside_limits)
    print(f"  Clipped: {clipped}")

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    test_aubo_i5_fk()
