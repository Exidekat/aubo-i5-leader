#!/usr/bin/env python3
"""Workspace Translation between SO-100 Leader and AUBO i5 Follower.

This module provides functions to translate end effector poses from the
SO-100 leader arm workspace to the AUBO i5 follower arm workspace.

The translation involves:
1. Position scaling (SO-100 has smaller workspace than AUBO i5)
2. Optional workspace offset/translation
3. Orientation mapping (preserving or adjusting orientations)
"""

import sys
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass

# Add external directory to path
sys.path.insert(0, str(Path(__file__).parent / "external"))

from forward_kinematics import SO100ForwardKinematics
from aubo_i5_kinematics import AuboI5ForwardKinematics


@dataclass
class WorkspaceInfo:
    """Workspace characteristics for a robot arm.

    Attributes:
        max_reach: Maximum reach distance from base (meters)
        min_reach: Minimum reach distance from base (meters)
        base_height: Height of robot base (meters)
        workspace_center: Typical center of workspace [x, y, z] (meters)
    """
    max_reach: float
    min_reach: float
    base_height: float
    workspace_center: np.ndarray


class WorkspaceTranslator:
    """Translates poses between SO-100 and AUBO i5 workspaces.

    The AUBO i5 has a larger workspace than the SO-100, so positions
    are scaled up when translating from leader to follower.
    """

    # SO-100 workspace characteristics (estimated from link lengths)
    # Link lengths: ~0.045m + 0.103m + 0.113m + 0.135m + 0.06m ≈ 0.456m
    SO100_MAX_REACH = 0.45  # meters (conservative estimate)
    SO100_MIN_REACH = 0.05  # meters
    SO100_BASE_HEIGHT = 0.0165  # meters

    # AUBO i5 workspace characteristics (from specifications)
    # Upper arm (408mm) + Forearm (376mm) + wrist offsets ≈ 886.5mm
    AUBO_I5_MAX_REACH = 0.8865  # meters (from spec)
    AUBO_I5_MIN_REACH = 0.10  # meters
    AUBO_I5_BASE_HEIGHT = 0.122  # meters

    def __init__(
        self,
        scale_factor: Optional[float] = None,
        workspace_offset: Optional[np.ndarray] = None,
        preserve_orientation: bool = True
    ):
        """Initialize workspace translator.

        Args:
            scale_factor: Custom scaling factor (auto-computed if None)
            workspace_offset: Translation offset [x, y, z] in AUBO workspace (meters)
            preserve_orientation: If True, preserve SO-100 orientation;
                                 if False, use identity orientation
        """
        # Compute scale factor based on max reach ratio
        if scale_factor is None:
            self.scale_factor = self.AUBO_I5_MAX_REACH / self.SO100_MAX_REACH
        else:
            self.scale_factor = scale_factor

        # Workspace offset in AUBO i5 frame
        if workspace_offset is None:
            # Default: no additional offset (just scaling)
            self.workspace_offset = np.array([0.0, 0.0, 0.0])
        else:
            self.workspace_offset = workspace_offset

        self.preserve_orientation = preserve_orientation

        # Initialize FK solvers for workspace analysis
        self.so100_fk = SO100ForwardKinematics()
        self.aubo_fk = AuboI5ForwardKinematics()

        print(f"Workspace Translator initialized:")
        print(f"  Scale factor: {self.scale_factor:.3f}x")
        print(f"  Workspace offset: {self.workspace_offset}")
        print(f"  Preserve orientation: {self.preserve_orientation}")

    def get_so100_workspace_info(self) -> WorkspaceInfo:
        """Get SO-100 workspace information.

        Returns:
            WorkspaceInfo for SO-100
        """
        return WorkspaceInfo(
            max_reach=self.SO100_MAX_REACH,
            min_reach=self.SO100_MIN_REACH,
            base_height=self.SO100_BASE_HEIGHT,
            workspace_center=np.array([0.0, 0.2, 0.15])  # Typical working area
        )

    def get_aubo_i5_workspace_info(self) -> WorkspaceInfo:
        """Get AUBO i5 workspace information.

        Returns:
            WorkspaceInfo for AUBO i5
        """
        return WorkspaceInfo(
            max_reach=self.AUBO_I5_MAX_REACH,
            min_reach=self.AUBO_I5_MIN_REACH,
            base_height=self.AUBO_I5_BASE_HEIGHT,
            workspace_center=np.array([0.0, 0.4, 0.3])  # Scaled working area
        )

    def translate_pose(
        self,
        so100_position: np.ndarray,
        so100_rotation: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Translate SO-100 end effector pose to AUBO i5 workspace.

        Args:
            so100_position: SO-100 EE position [x, y, z] in meters
            so100_rotation: SO-100 EE rotation matrix [3x3]

        Returns:
            Tuple of (aubo_position, aubo_rotation) in AUBO i5 workspace
        """
        # Scale position
        aubo_position = so100_position * self.scale_factor

        # Add workspace offset
        aubo_position = aubo_position + self.workspace_offset

        # Handle orientation
        if self.preserve_orientation:
            aubo_rotation = so100_rotation.copy()
        else:
            # Use identity orientation (pointing down, typical for tabletop tasks)
            aubo_rotation = np.eye(3)

        return aubo_position, aubo_rotation

    def translate_position_only(self, so100_position: np.ndarray) -> np.ndarray:
        """Translate only position (no orientation).

        Args:
            so100_position: SO-100 EE position [x, y, z] in meters

        Returns:
            AUBO i5 EE position [x, y, z] in meters
        """
        aubo_position = so100_position * self.scale_factor
        aubo_position = aubo_position + self.workspace_offset
        return aubo_position

    def is_aubo_pose_reachable(
        self,
        aubo_position: np.ndarray,
        safety_margin: float = 0.05
    ) -> bool:
        """Check if translated pose is reachable by AUBO i5.

        Args:
            aubo_position: Target position in AUBO workspace
            safety_margin: Safety margin from max reach (meters)

        Returns:
            True if position is within reachable workspace
        """
        distance_from_base = np.linalg.norm(aubo_position)

        # Check if within reach limits (with safety margin)
        max_safe_reach = self.AUBO_I5_MAX_REACH - safety_margin
        min_safe_reach = self.AUBO_I5_MIN_REACH + safety_margin

        return min_safe_reach <= distance_from_base <= max_safe_reach

    def print_workspace_comparison(self):
        """Print detailed workspace comparison."""
        so100_info = self.get_so100_workspace_info()
        aubo_info = self.get_aubo_i5_workspace_info()

        print("\n" + "="*70)
        print("Workspace Comparison: SO-100 Leader vs AUBO i5 Follower")
        print("="*70)

        print("\nSO-100 Leader:")
        print(f"  Max reach:        {so100_info.max_reach*1000:.1f} mm")
        print(f"  Min reach:        {so100_info.min_reach*1000:.1f} mm")
        print(f"  Base height:      {so100_info.base_height*1000:.1f} mm")
        print(f"  Workspace center: {so100_info.workspace_center*1000} mm")

        print("\nAUBO i5 Follower:")
        print(f"  Max reach:        {aubo_info.max_reach*1000:.1f} mm")
        print(f"  Min reach:        {aubo_info.min_reach*1000:.1f} mm")
        print(f"  Base height:      {aubo_info.base_height*1000:.1f} mm")
        print(f"  Workspace center: {aubo_info.workspace_center*1000} mm")

        print("\nTranslation Parameters:")
        print(f"  Scale factor:     {self.scale_factor:.3f}x")
        print(f"  Workspace offset: {self.workspace_offset*1000} mm")
        print(f"  Preserve orient:  {self.preserve_orientation}")

        print("\n" + "="*70)


def test_workspace_translation():
    """Test workspace translation with sample SO-100 poses."""

    # Initialize translator
    translator = WorkspaceTranslator()
    translator.print_workspace_comparison()

    # Initialize FK solvers
    so100_fk = SO100ForwardKinematics()

    print("\n" + "="*70)
    print("Workspace Translation Tests")
    print("="*70)

    # Test 1: SO-100 home position
    print("\nTest 1: SO-100 home position")
    so100_angles_1 = np.zeros(6)
    so100_pos_1, so100_rot_1 = so100_fk.compute_end_effector_pose(so100_angles_1)

    print(f"  SO-100 position: {so100_pos_1}")
    print(f"  SO-100 distance: {np.linalg.norm(so100_pos_1):.4f} m")

    aubo_pos_1, aubo_rot_1 = translator.translate_pose(so100_pos_1, so100_rot_1)
    print(f"  AUBO i5 position: {aubo_pos_1}")
    print(f"  AUBO i5 distance: {np.linalg.norm(aubo_pos_1):.4f} m")
    print(f"  Reachable: {translator.is_aubo_pose_reachable(aubo_pos_1)}")

    # Test 2: SO-100 extended forward
    print("\nTest 2: SO-100 extended forward")
    so100_angles_2 = np.array([0.0, -np.pi/4, np.pi/2, -np.pi/4, 0.0, 0.0])
    so100_pos_2, so100_rot_2 = so100_fk.compute_end_effector_pose(so100_angles_2)

    print(f"  SO-100 position: {so100_pos_2}")
    print(f"  SO-100 distance: {np.linalg.norm(so100_pos_2):.4f} m")

    aubo_pos_2, aubo_rot_2 = translator.translate_pose(so100_pos_2, so100_rot_2)
    print(f"  AUBO i5 position: {aubo_pos_2}")
    print(f"  AUBO i5 distance: {np.linalg.norm(aubo_pos_2):.4f} m")
    print(f"  Reachable: {translator.is_aubo_pose_reachable(aubo_pos_2)}")

    # Test 3: SO-100 folded configuration
    print("\nTest 3: SO-100 folded configuration")
    so100_angles_3 = np.array([np.pi/4, np.pi/4, np.pi/4, 0.0, 0.0, 0.0])
    so100_pos_3, so100_rot_3 = so100_fk.compute_end_effector_pose(so100_angles_3)

    print(f"  SO-100 position: {so100_pos_3}")
    print(f"  SO-100 distance: {np.linalg.norm(so100_pos_3):.4f} m")

    aubo_pos_3, aubo_rot_3 = translator.translate_pose(so100_pos_3, so100_rot_3)
    print(f"  AUBO i5 position: {aubo_pos_3}")
    print(f"  AUBO i5 distance: {np.linalg.norm(aubo_pos_3):.4f} m")
    print(f"  Reachable: {translator.is_aubo_pose_reachable(aubo_pos_3)}")

    # Test 4: Custom workspace offset
    print("\nTest 4: Custom workspace offset (shift AUBO workspace)")
    translator_offset = WorkspaceTranslator(
        workspace_offset=np.array([0.0, 0.1, 0.0])  # 10cm Y offset
    )

    aubo_pos_4, aubo_rot_4 = translator_offset.translate_pose(so100_pos_2, so100_rot_2)
    print(f"  SO-100 position: {so100_pos_2}")
    print(f"  AUBO i5 position (with offset): {aubo_pos_4}")
    print(f"  Reachable: {translator_offset.is_aubo_pose_reachable(aubo_pos_4)}")

    print("\n" + "="*70)
    print("All workspace translation tests completed!")
    print("="*70)


if __name__ == "__main__":
    test_workspace_translation()
