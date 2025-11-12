#!/usr/bin/env python3
"""SO-100 Leader to AUBO i5 Follower Control System.

This script implements a leader-follower robot control system where:
1. SO-100 arm acts as the leader (reads joint angles via serial)
2. Forward kinematics computes SO-100 end effector pose
3. Workspace translation scales/translates pose to AUBO i5 workspace
4. Inverse kinematics solves AUBO i5 joint angles
5. Visualization shows both arms in real-time (or sends to AUBO i5)

Usage:
    # Test mode (simulated SO-100 motion)
    python leader_follower.py --test

    # Real mode (with SO-100 hardware)
    python leader_follower.py --robot-id so100_leader --port /dev/ttyACM0

    # With custom workspace offset
    python leader_follower.py --test --offset 0.0 0.1 0.0

    # Control AUBO i5 (requires pyaubo-sdk and connection)
    python leader_follower.py --test --control-aubo --aubo-ip 192.168.1.100
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

# Add external directory to path
sys.path.insert(0, str(Path(__file__).parent / "external"))

from so100_direct import DirectSO100
from forward_kinematics import SO100ForwardKinematics
from aubo_i5_kinematics import AuboI5ForwardKinematics
from aubo_i5_ik import AuboI5InverseKinematics
from workspace_translation import WorkspaceTranslator


class LeaderFollowerController:
    """Coordinates SO-100 leader and AUBO i5 follower robot control."""

    def __init__(
        self,
        workspace_offset: Optional[np.ndarray] = None,
        scale_factor: Optional[float] = None,
        preserve_orientation: bool = False,  # Default: use consistent downward orientation
        ik_damping: float = 0.1,
        verbose: bool = False
    ):
        """Initialize leader-follower controller.

        Args:
            workspace_offset: Translation offset for AUBO workspace
            scale_factor: Custom scaling factor (auto if None)
            preserve_orientation: Preserve SO-100 orientation or use identity
            ik_damping: Damping factor for IK solver
            verbose: Print detailed status messages
        """
        self.verbose = verbose

        # Initialize kinematics
        if self.verbose:
            print("Initializing kinematics solvers...")

        self.so100_fk = SO100ForwardKinematics()
        self.aubo_fk = AuboI5ForwardKinematics()
        self.aubo_ik = AuboI5InverseKinematics(self.aubo_fk)
        self.ik_damping = ik_damping

        # Initialize workspace translator
        if self.verbose:
            print("Initializing workspace translator...")

        self.translator = WorkspaceTranslator(
            scale_factor=scale_factor,
            workspace_offset=workspace_offset,
            preserve_orientation=preserve_orientation
        )

        # State tracking
        self.current_so100_joints = np.zeros(6)
        self.current_aubo_joints = np.zeros(6)
        self.last_aubo_ik_converged = False
        self.ik_failures = 0
        self.total_updates = 0

        if self.verbose:
            print("✓ Leader-follower controller initialized\n")

    def update(self, so100_joint_angles: np.ndarray) -> Tuple[np.ndarray, bool, dict]:
        """Update follower position based on leader joint angles.

        Args:
            so100_joint_angles: SO-100 joint angles [6D array, 5 joints + gripper]

        Returns:
            Tuple of:
                - aubo_joint_angles: Computed AUBO i5 joint angles (6D)
                - success: True if IK converged
                - info: Dictionary with detailed information
        """
        self.total_updates += 1

        # Step 1: Compute SO-100 end effector pose (forward kinematics)
        so100_pos, so100_rot = self.so100_fk.compute_end_effector_pose(so100_joint_angles)

        # Step 2: Translate pose to AUBO i5 workspace
        aubo_target_pos, aubo_target_rot = self.translator.translate_pose(
            so100_pos, so100_rot
        )

        # Check if target is reachable
        is_reachable = self.translator.is_aubo_pose_reachable(aubo_target_pos)

        if not is_reachable:
            if self.verbose:
                print(f"⚠ Warning: Target position {aubo_target_pos} may not be reachable")

        # Step 3: Solve inverse kinematics for AUBO i5
        # Use position-only IK for better convergence (orientation less critical)
        aubo_joints, converged, iterations, error = self.aubo_ik.solve_position_only(
            target_pos=aubo_target_pos,
            initial_guess=self.current_aubo_joints,  # Use previous solution as guess
            max_iterations=50,
            tolerance=1e-3,  # 1mm tolerance
            damping=self.ik_damping,
            verbose=False
        )

        # Update state
        self.current_so100_joints = so100_joint_angles
        if converged:
            self.current_aubo_joints = aubo_joints
            self.last_aubo_ik_converged = True
        else:
            self.ik_failures += 1
            self.last_aubo_ik_converged = False

        # Compile information
        info = {
            'so100_pos': so100_pos,
            'so100_rot': so100_rot,
            'aubo_target_pos': aubo_target_pos,
            'aubo_target_rot': aubo_target_rot,
            'is_reachable': is_reachable,
            'ik_converged': converged,
            'ik_iterations': iterations,
            'ik_error': error,
            'total_updates': self.total_updates,
            'ik_failures': self.ik_failures,
            'success_rate': (self.total_updates - self.ik_failures) / self.total_updates
        }

        return aubo_joints, converged, info

    def print_status(self, info: dict):
        """Print formatted status information.

        Args:
            info: Information dictionary from update()
        """
        print(f"\n{'='*70}")
        print(f"Leader-Follower Status (Update #{info['total_updates']})")
        print(f"{'='*70}")

        print(f"\nSO-100 Leader:")
        print(f"  Position: {info['so100_pos']}")
        print(f"  Distance: {np.linalg.norm(info['so100_pos']):.4f} m")

        print(f"\nAUBO i5 Follower Target:")
        print(f"  Position: {info['aubo_target_pos']}")
        print(f"  Distance: {np.linalg.norm(info['aubo_target_pos']):.4f} m")
        print(f"  Reachable: {'✓' if info['is_reachable'] else '✗'}")

        print(f"\nInverse Kinematics:")
        print(f"  Converged: {'✓' if info['ik_converged'] else '✗'}")
        print(f"  Iterations: {info['ik_iterations']}")
        print(f"  Error: {info['ik_error']:.6f} m")

        print(f"\nStatistics:")
        print(f"  Total updates: {info['total_updates']}")
        print(f"  IK failures: {info['ik_failures']}")
        print(f"  Success rate: {info['success_rate']*100:.1f}%")

        print(f"{'='*70}")


def test_mode(
    update_rate: float,
    workspace_offset: Optional[np.ndarray],
    scale_factor: Optional[float],
    duration: Optional[float]
):
    """Run in test mode with simulated SO-100 motion.

    Args:
        update_rate: Update frequency in Hz
        workspace_offset: Workspace translation offset
        scale_factor: Custom scaling factor
        duration: Test duration in seconds (None for infinite)
    """
    print("\n🧪 Running in TEST MODE")
    print("   Simulating SO-100 leader with sinusoidal motion\n")

    # Initialize controller
    controller = LeaderFollowerController(
        workspace_offset=workspace_offset,
        scale_factor=scale_factor,
        preserve_orientation=False,
        verbose=True
    )

    # Print workspace info
    controller.translator.print_workspace_comparison()

    print("\n" + "="*70)
    print("Starting leader-follower simulation...")
    print("Press Ctrl+C to stop")
    print("="*70)

    t = 0.0
    dt = 1.0 / update_rate
    start_time = time.time()
    last_print_time = 0.0

    try:
        while True:
            loop_start = time.time()

            # Generate simulated SO-100 joint angles (5 joints + gripper)
            so100_joints = np.array([
                0.3 * np.sin(t * 0.5),         # Shoulder_Rotation
                0.4 * np.sin(t * 0.4),         # Shoulder_Pitch
                0.5 * np.cos(t * 0.3),         # Elbow
                0.3 * np.sin(t * 0.6),         # Wrist_Pitch
                0.2 * np.cos(t * 0.7),         # Wrist_Roll
                0.0                            # Gripper
            ])

            # Update follower
            aubo_joints, success, info = controller.update(so100_joints)

            # Print status every second
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                controller.print_status(info)
                print(f"\nComputed AUBO i5 joint angles:")
                print(f"  {np.rad2deg(aubo_joints)}")
                last_print_time = current_time

            # Check duration limit
            if duration is not None and (current_time - start_time) >= duration:
                print(f"\n✓ Test duration ({duration}s) reached")
                break

            # Maintain update rate
            t += dt
            loop_duration = time.time() - loop_start
            sleep_time = max(0, dt - loop_duration)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n⏹️  Test stopped by user")

    # Final statistics
    print("\n" + "="*70)
    print("Test Mode Final Statistics")
    print("="*70)
    print(f"Total updates: {controller.total_updates}")
    print(f"IK failures: {controller.ik_failures}")
    print(f"Success rate: {(controller.total_updates - controller.ik_failures) / controller.total_updates * 100:.1f}%")
    print("="*70)


def real_mode(
    robot_id: str,
    port: str,
    update_rate: float,
    workspace_offset: Optional[np.ndarray],
    scale_factor: Optional[float],
    control_aubo: bool,
    aubo_ip: Optional[str]
):
    """Run with real SO-100 hardware.

    Args:
        robot_id: SO-100 robot ID
        port: Serial port
        update_rate: Update frequency in Hz
        workspace_offset: Workspace translation offset
        scale_factor: Custom scaling factor
        control_aubo: If True, send commands to AUBO i5
        aubo_ip: AUBO i5 IP address (required if control_aubo=True)
    """
    print(f"\n🤖 Running in REAL MODE")
    print(f"   SO-100 Robot ID: {robot_id}")
    print(f"   Serial Port: {port}")

    if control_aubo:
        if aubo_ip is None:
            print("\n❌ Error: --aubo-ip required when --control-aubo is set")
            return
        print(f"   AUBO i5 IP: {aubo_ip}")
        print("   ⚠ AUBO i5 control not yet implemented (visualization only)")
    else:
        print("   AUBO i5: Visualization only (no hardware control)")

    print()

    # Initialize controller
    controller = LeaderFollowerController(
        workspace_offset=workspace_offset,
        scale_factor=scale_factor,
        preserve_orientation=False,
        verbose=True
    )

    # Connect to SO-100
    print("Connecting to SO-100...")
    robot = DirectSO100(robot_id=robot_id, port=port)
    if not robot.connect():
        print("\n❌ Failed to connect to SO-100!")
        return

    print("✓ Connected to SO-100\n")

    # Print workspace info
    controller.translator.print_workspace_comparison()

    print("\n" + "="*70)
    print("Reading SO-100 and computing AUBO i5 joint angles...")
    print("Press Ctrl+C to stop")
    print("="*70)

    dt = 1.0 / update_rate
    last_print_time = 0.0

    try:
        while True:
            loop_start = time.time()

            # Read SO-100 joint positions
            so100_joints = robot.get_joint_positions()

            # Update follower
            aubo_joints, success, info = controller.update(so100_joints)

            # Print status every second
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                controller.print_status(info)
                print(f"\nSO-100 joint angles (deg):")
                print(f"  {np.rad2deg(so100_joints[:5])}")  # First 5 joints only
                print(f"\nComputed AUBO i5 joint angles (deg):")
                print(f"  {np.rad2deg(aubo_joints)}")
                last_print_time = current_time

            # TODO: Send aubo_joints to AUBO i5 if control_aubo=True
            # Would require pyaubo-sdk integration here

            # Maintain update rate
            loop_duration = time.time() - loop_start
            sleep_time = max(0, dt - loop_duration)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user")
    finally:
        robot.disconnect()
        print("✓ Disconnected from SO-100")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SO-100 Leader to AUBO i5 Follower Control System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test mode with default settings
  python leader_follower.py --test

  # Test mode with custom workspace offset (10cm in Y direction)
  python leader_follower.py --test --offset 0.0 0.1 0.0

  # Test mode with custom scale factor
  python leader_follower.py --test --scale 2.5

  # Real mode with SO-100 hardware
  python leader_follower.py --robot-id so100_leader --port /dev/ttyACM0

  # Real mode with AUBO i5 control (requires SDK)
  python leader_follower.py --robot-id so100_leader --port /dev/ttyACM0 \\
      --control-aubo --aubo-ip 192.168.1.100
        """
    )

    # Mode selection
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with simulated SO-100 motion"
    )

    # SO-100 connection (real mode)
    parser.add_argument("--robot-id", type=str, help="SO-100 robot ID")
    parser.add_argument("--port", type=str, help="Serial port for SO-100")

    # Workspace translation
    parser.add_argument(
        "--offset",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Workspace offset [x y z] in meters"
    )
    parser.add_argument(
        "--scale",
        type=float,
        help="Custom scaling factor (default: auto-computed)"
    )

    # Update rate
    parser.add_argument(
        "--rate",
        type=float,
        default=10.0,
        help="Update rate in Hz (default: 10)"
    )

    # Test mode duration
    parser.add_argument(
        "--duration",
        type=float,
        help="Test duration in seconds (default: infinite)"
    )

    # AUBO i5 control
    parser.add_argument(
        "--control-aubo",
        action="store_true",
        help="Send commands to AUBO i5 (requires pyaubo-sdk)"
    )
    parser.add_argument(
        "--aubo-ip",
        type=str,
        help="AUBO i5 IP address (required if --control-aubo)"
    )

    args = parser.parse_args()

    # Parse workspace offset
    workspace_offset = None
    if args.offset:
        workspace_offset = np.array(args.offset)

    # Validate arguments
    if not args.test and (not args.robot_id or not args.port):
        parser.error("--robot-id and --port required unless --test is specified")

    # Run appropriate mode
    if args.test:
        test_mode(
            update_rate=args.rate,
            workspace_offset=workspace_offset,
            scale_factor=args.scale,
            duration=args.duration
        )
    else:
        real_mode(
            robot_id=args.robot_id,
            port=args.port,
            update_rate=args.rate,
            workspace_offset=workspace_offset,
            scale_factor=args.scale,
            control_aubo=args.control_aubo,
            aubo_ip=args.aubo_ip
        )


if __name__ == "__main__":
    main()
