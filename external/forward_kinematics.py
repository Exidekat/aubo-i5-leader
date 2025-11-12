"""Forward kinematics implementation for robot arms using URDF transforms."""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class URDFJoint:
    """URDF-style joint definition.

    Attributes:
        name: Joint name
        origin_xyz: Translation [x, y, z] in meters
        origin_rpy: Rotation [roll, pitch, yaw] in radians
        axis: Rotation axis [x, y, z] (normalized)
        joint_type: 'revolute' or 'fixed'
    """
    name: str
    origin_xyz: np.ndarray  # [x, y, z]
    origin_rpy: np.ndarray  # [roll, pitch, yaw]
    axis: np.ndarray        # [x, y, z]
    joint_type: str = 'revolute'


class ForwardKinematics:
    """Base class for forward kinematics computation using URDF transforms."""

    def __init__(self, joints: List[URDFJoint]):
        """Initialize forward kinematics with URDF joint definitions.

        Args:
            joints: List of URDF joint definitions
        """
        self.joints = joints
        self.num_joints = len(joints)

    @staticmethod
    def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert roll-pitch-yaw to 3x3 rotation matrix.

        Args:
            roll: Rotation about X-axis (radians)
            pitch: Rotation about Y-axis (radians)
            yaw: Rotation about Z-axis (radians)

        Returns:
            3x3 rotation matrix
        """
        # Roll (X-axis rotation)
        cr, sr = np.cos(roll), np.sin(roll)
        Rx = np.array([
            [1, 0, 0],
            [0, cr, -sr],
            [0, sr, cr]
        ])

        # Pitch (Y-axis rotation)
        cp, sp = np.cos(pitch), np.sin(pitch)
        Ry = np.array([
            [cp, 0, sp],
            [0, 1, 0],
            [-sp, 0, cp]
        ])

        # Yaw (Z-axis rotation)
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rz = np.array([
            [cy, -sy, 0],
            [sy, cy, 0],
            [0, 0, 1]
        ])

        # Combined rotation: R = Rz * Ry * Rx
        return Rz @ Ry @ Rx

    @staticmethod
    def axis_angle_to_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
        """Convert axis-angle to 3x3 rotation matrix using Rodrigues' formula.

        Args:
            axis: Rotation axis [x, y, z] (must be normalized)
            angle: Rotation angle in radians

        Returns:
            3x3 rotation matrix
        """
        axis = axis / np.linalg.norm(axis)  # Ensure normalized
        x, y, z = axis
        c, s = np.cos(angle), np.sin(angle)
        C = 1 - c

        return np.array([
            [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
            [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
            [z*x*C - y*s, z*y*C + x*s, z*z*C + c]
        ])

    def compute_joint_transform(self, joint: URDFJoint, joint_angle: float) -> np.ndarray:
        """Compute 4x4 transformation matrix for a joint.

        Args:
            joint: URDF joint definition
            joint_angle: Joint angle in radians (for revolute joints)

        Returns:
            4x4 homogeneous transformation matrix
        """
        # Start with origin transform (xyz + rpy)
        T = np.eye(4)
        T[:3, :3] = self.rpy_to_rotation_matrix(*joint.origin_rpy)
        T[:3, 3] = joint.origin_xyz

        # Add joint rotation if revolute
        # The axis is specified in the joint frame (after origin transform)
        if joint.joint_type == 'revolute':
            R_joint = self.axis_angle_to_rotation_matrix(joint.axis, joint_angle)
            T[:3, :3] = T[:3, :3] @ R_joint

        return T

    def compute_joint_positions(self, joint_angles: np.ndarray) -> List[np.ndarray]:
        """Compute 3D position of each joint given joint angles.

        Args:
            joint_angles: Joint angles in radians (length = num_joints)

        Returns:
            List of 3D positions [x, y, z] for each joint (including base at origin)
        """
        if len(joint_angles) != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joint angles, got {len(joint_angles)}")

        # Global rotation: -90° about X axis (clockwise when looking along +X)
        # This rotates the entire robot FK to match visualization expectations
        R_global = self.rpy_to_rotation_matrix(-np.pi/2, 0, 0)
        T_global = np.eye(4)
        T_global[:3, :3] = R_global

        positions = [np.array([0.0, 0.0, 0.0])]  # Base at origin

        T = T_global  # Start with global rotation

        for i, (joint, angle) in enumerate(zip(self.joints, joint_angles)):
            # Compute transformation for this joint
            T_i = self.compute_joint_transform(joint, angle)

            # Update cumulative transformation
            T = T @ T_i

            # Extract position from transformation matrix
            position = T[:3, 3]
            positions.append(position.copy())

        return positions

    def compute_joint_frames(self, joint_angles: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """Compute coordinate frame at each joint (position + rotation axes).

        For revolute joints, the rotation axis is defined by the joint's axis parameter.

        Args:
            joint_angles: Joint angles in radians (length = num_joints)

        Returns:
            List of frames, each containing:
                - 'position': [x, y, z] position
                - 'rotation': 3x3 rotation matrix (columns are X, Y, Z axes)
                - 'z_axis': [x, y, z] rotation axis (actual joint axis transformed to world frame)
        """
        if len(joint_angles) != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joint angles, got {len(joint_angles)}")

        # Global rotation: -90° about X axis (clockwise when looking along +X)
        # This rotates the entire robot FK to match visualization expectations
        R_global = self.rpy_to_rotation_matrix(-np.pi/2, 0, 0)
        T_global = np.eye(4)
        T_global[:3, :3] = R_global

        frames = [
            {
                'position': np.array([0.0, 0.0, 0.0]),
                'rotation': R_global.copy(),  # Apply global rotation to base frame
                'z_axis': R_global @ np.array([0.0, 0.0, 1.0])  # Rotated base Z axis
            }
        ]

        T = T_global  # Start with global rotation

        for i, (joint, angle) in enumerate(zip(self.joints, joint_angles)):
            # Compute transformation for this joint
            T_i = self.compute_joint_transform(joint, angle)

            # Update cumulative transformation
            T = T @ T_i

            # Extract position and rotation
            position = T[:3, 3]
            rotation = T[:3, :3]

            # Transform joint axis to world frame
            z_axis = rotation @ joint.axis

            frames.append({
                'position': position.copy(),
                'rotation': rotation.copy(),
                'z_axis': z_axis.copy()
            })

        return frames

    def compute_end_effector_pose(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute end effector position and orientation.

        Args:
            joint_angles: Joint angles in radians

        Returns:
            Tuple of (position [x,y,z], rotation_matrix [3x3])
        """
        # Global rotation: -90° about X axis (clockwise when looking along +X)
        # This rotates the entire robot FK to match visualization expectations
        R_global = self.rpy_to_rotation_matrix(-np.pi/2, 0, 0)
        T_global = np.eye(4)
        T_global[:3, :3] = R_global

        T = T_global  # Start with global rotation

        for joint, angle in zip(self.joints, joint_angles):
            T_i = self.compute_joint_transform(joint, angle)
            T = T @ T_i

        position = T[:3, 3]
        rotation = T[:3, :3]

        return position, rotation


class SO100ForwardKinematics(ForwardKinematics):
    """Forward kinematics for SO-100 robotic arm using URDF transforms.

    SO-100 is a 5-DOF arm with 1-DOF gripper (6 joints total):
    - Joint 1: Shoulder Rotation (base pan, Y-axis)
    - Joint 2: Shoulder Pitch (shoulder lift, X-axis)
    - Joint 3: Elbow (elbow flex, X-axis)
    - Joint 4: Wrist Pitch (wrist flex, X-axis)
    - Joint 5: Wrist Roll (wrist rotation, Y-axis)
    - Joint 6: Gripper (jaw open/close, Z-axis)

    URDF source: https://github.com/brukg/SO-100-arm/blob/main/urdf/so_100_arm_5dof.urdf

    Joint Definitions Table:
    ┌──────────────────────┬─────────────────────┬────────────┬─────────────────────┬────────────────┐
    │ Joint Name           │ Origin XYZ (m)      │ Origin RPY │ Axis               │ Parent Link    │
    ├──────────────────────┼─────────────────────┼────────────┼─────────────────────┼────────────────┤
    │ Shoulder_Rotation    │ [0, -0.0452, 0.0165] │ [π/2, 0, 0] │ [0, 1, 0] (Y-axis) │ Base           │
    │ Shoulder_Pitch       │ [0, 0.1025, 0.0306]  │ [0, 0, 0]   │ [1, 0, 0] (X-axis) │ Shoulder_Rot   │
    │ Elbow                │ [0, 0.11257, 0.028]  │ [0, 0, 0]   │ [1, 0, 0] (X-axis) │ Upper_Arm      │
    │ Wrist_Pitch          │ [0, 0.0052, 0.1349]  │ [0, 0, 0]   │ [1, 0, 0] (X-axis) │ Lower_Arm      │
    │ Wrist_Roll           │ [0, -0.0601, 0]      │ [0, 0, 0]   │ [0, 1, 0] (Y-axis) │ Wrist_Pitch_R  │
    │ Gripper              │ [-0.0202, -0.0244, 0] │ [π, 0, π]   │ [0, 0, 1] (Z-axis) │ Fixed_Gripper  │
    └──────────────────────┴─────────────────────┴────────────┴─────────────────────┴────────────────┘

    All measurements are in meters and radians.
    """

    def __init__(self):
        """Initialize SO-100 forward kinematics with URDF joint definitions."""
        # Joint definitions from official SO-100 URDF
        joints = [
            # Joint 1: Shoulder Rotation (base pan around Y-axis)
            URDFJoint(
                name='Shoulder_Rotation',
                origin_xyz=np.array([0.0, -0.0452, 0.0165]),
                origin_rpy=np.array([np.pi/2, 0.0, 0.0]),  # 90° roll
                axis=np.array([0.0, 1.0, 0.0]),  # Y-axis rotation
                joint_type='revolute'
            ),

            # Joint 2: Shoulder Pitch (shoulder lift around X-axis)
            URDFJoint(
                name='Shoulder_Pitch',
                origin_xyz=np.array([0.0, 0.1025, 0.0306]),
                origin_rpy=np.array([0.0, 0.0, 0.0]),
                axis=np.array([1.0, 0.0, 0.0]),  # X-axis rotation
                joint_type='revolute'
            ),

            # Joint 3: Elbow (elbow flex around X-axis)
            URDFJoint(
                name='Elbow',
                origin_xyz=np.array([0.0, 0.11257, 0.028]),
                origin_rpy=np.array([0.0, 0.0, 0.0]),
                axis=np.array([1.0, 0.0, 0.0]),  # X-axis rotation
                joint_type='revolute'
            ),

            # Joint 4: Wrist Pitch (wrist flex around X-axis)
            URDFJoint(
                name='Wrist_Pitch',
                origin_xyz=np.array([0.0, 0.0052, 0.1349]),
                origin_rpy=np.array([0.0, 0.0, 0.0]),
                axis=np.array([1.0, 0.0, 0.0]),  # X-axis rotation
                joint_type='revolute'
            ),

            # Joint 5: Wrist Roll (wrist rotation around Y-axis)
            URDFJoint(
                name='Wrist_Roll',
                origin_xyz=np.array([0.0, -0.0601, 0.0]),
                origin_rpy=np.array([0.0, 0.0, 0.0]),
                axis=np.array([0.0, 1.0, 0.0]),  # Y-axis rotation
                joint_type='revolute'
            ),

            # Joint 6: Gripper (jaw open/close around Z-axis)
            URDFJoint(
                name='Gripper',
                origin_xyz=np.array([-0.0202, -0.0244, 0.0]),
                origin_rpy=np.array([np.pi, 0.0, np.pi]),  # 180° roll and yaw
                axis=np.array([0.0, 0.0, 1.0]),  # Z-axis rotation
                joint_type='revolute'
            ),
        ]

        super().__init__(joints)

    def compute_with_gripper(self, joint_angles: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute FK including gripper state.

        Args:
            joint_angles: 6D array [6 joint angles including gripper]

        Returns:
            Dictionary with:
                - 'joint_positions': List of 3D positions for each joint
                - 'joint_frames': List of coordinate frames (position + rotation axes)
                - 'end_effector_pos': End effector position [x, y, z]
                - 'end_effector_rot': End effector rotation matrix [3x3]
                - 'gripper_state': Gripper opening (0=closed, 1=open)
        """
        if len(joint_angles) != 6:
            raise ValueError(f"SO-100 requires 6 joint angles, got {len(joint_angles)}")

        # Compute FK for all 6 joints (including gripper)
        joint_positions = self.compute_joint_positions(joint_angles)
        joint_frames = self.compute_joint_frames(joint_angles)

        ee_pos, ee_rot = self.compute_end_effector_pose(joint_angles)

        gripper_state = float(joint_angles[5])

        return {
            'joint_positions': joint_positions,
            'joint_frames': joint_frames,
            'end_effector_pos': ee_pos,
            'end_effector_rot': ee_rot,
            'gripper_state': gripper_state,
            'joint_angles': joint_angles,
        }


def test_so100_fk():
    """Test SO-100 forward kinematics with sample configurations."""
    fk = SO100ForwardKinematics()

    # Test 1: Home position (all zeros)
    print("Test 1: Home position")
    home = np.zeros(6)
    result = fk.compute_with_gripper(home)
    print(f"  End effector: {result['end_effector_pos']}")
    print(f"  Gripper: {result['gripper_state']}")

    # Test 2: Arm extended forward
    print("\nTest 2: Arm extended forward")
    extended = np.array([0.0, -np.pi/4, np.pi/2, -np.pi/4, 0.0, 0.5])
    result = fk.compute_with_gripper(extended)
    print(f"  End effector: {result['end_effector_pos']}")
    print(f"  Num joints: {len(result['joint_positions'])}")

    # Test 3: Arm bent upward
    print("\nTest 3: Arm bent upward")
    upward = np.array([0.0, np.pi/4, np.pi/4, 0.0, 0.0, 1.0])
    result = fk.compute_with_gripper(upward)
    print(f"  End effector: {result['end_effector_pos']}")


if __name__ == "__main__":
    test_so100_fk()
