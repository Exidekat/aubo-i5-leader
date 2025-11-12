#!/usr/bin/env python3
"""SO-100 Leader Arm Forward Kinematics with Visualization.

This script reads joint positions from the SO-100 leader arm via serial,
computes forward kinematics to determine the end effector position and
orientation, and displays the arm in a 3D Flask visualization.

Usage:
    python so100_leader_fk.py --robot-id <robot_id> --port <serial_port>

    # Example:
    python so100_leader_fk.py --robot-id so100_leader --port /dev/ttyACM0

    # Test mode (no hardware):
    python so100_leader_fk.py --test
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

# Add ltx_vla submodule to path
sys.path.insert(0, str(Path(__file__).parent / "submodules" / "ltx_vla"))

from ltx_vla.robots.so100_direct import DirectSO100
from ltx_vla.visualization.forward_kinematics import SO100ForwardKinematics
from ltx_vla.visualization.viz_server import RobotVizServer


def rotation_matrix_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """Convert 3x3 rotation matrix to Euler angles (roll, pitch, yaw).

    Args:
        R: 3x3 rotation matrix

    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    # Assuming ZYX convention
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)

    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return roll, pitch, yaw


def print_robot_state(joint_angles: np.ndarray, ee_pos: np.ndarray, ee_rot: np.ndarray):
    """Print formatted robot state information.

    Args:
        joint_angles: 6D array of joint angles in radians
        ee_pos: End effector position [x, y, z] in meters
        ee_rot: End effector rotation matrix [3x3]
    """
    # Convert rotation matrix to Euler angles
    roll, pitch, yaw = rotation_matrix_to_euler(ee_rot)

    print("\n" + "="*70)
    print("SO-100 Leader Arm State")
    print("="*70)

    # Joint angles (first 5 joints, ignoring gripper)
    joint_names = [
        "Shoulder_Rotation",
        "Shoulder_Pitch",
        "Elbow",
        "Wrist_Pitch",
        "Wrist_Roll"
    ]

    print("\nJoint Angles (rad):")
    for i, name in enumerate(joint_names):
        deg = np.rad2deg(joint_angles[i])
        print(f"  {name:18s}: {joint_angles[i]:+7.4f} rad  ({deg:+7.2f}°)")

    # End effector position
    print(f"\nEnd Effector Position (m):")
    print(f"  X: {ee_pos[0]:+7.4f}")
    print(f"  Y: {ee_pos[1]:+7.4f}")
    print(f"  Z: {ee_pos[2]:+7.4f}")
    print(f"  Distance from base: {np.linalg.norm(ee_pos):.4f} m")

    # End effector orientation
    print(f"\nEnd Effector Orientation (rad):")
    print(f"  Roll:  {roll:+7.4f} rad  ({np.rad2deg(roll):+7.2f}°)")
    print(f"  Pitch: {pitch:+7.4f} rad  ({np.rad2deg(pitch):+7.2f}°)")
    print(f"  Yaw:   {yaw:+7.4f} rad  ({np.rad2deg(yaw):+7.2f}°)")

    print("="*70)


def test_mode(update_rate: float = 10.0):
    """Run in test mode with simulated joint angles.

    Args:
        update_rate: Update frequency in Hz
    """
    print("\n🧪 Running in TEST MODE (no hardware)")
    print("   Simulating SO-100 joint angles with sinusoidal motion\n")

    # Initialize FK and visualization
    fk = SO100ForwardKinematics()
    viz_server = RobotVizServer(port=5000)
    viz_server.set_mode("demo")
    viz_server.start(blocking=False)

    print(f"\n🌐 Visualization: http://localhost:5000")
    print("\nPress Ctrl+C to stop\n")

    t = 0.0
    dt = 1.0 / update_rate

    try:
        while True:
            # Generate sinusoidal joint motion (5 joints + gripper)
            joint_angles = np.array([
                0.3 * np.sin(t * 0.5),         # Shoulder_Rotation
                0.4 * np.sin(t * 0.4),         # Shoulder_Pitch
                0.5 * np.cos(t * 0.3),         # Elbow
                0.3 * np.sin(t * 0.6),         # Wrist_Pitch
                0.2 * np.cos(t * 0.7),         # Wrist_Roll
                0.0                            # Gripper (not used)
            ])

            # Compute forward kinematics
            ee_pos, ee_rot = fk.compute_end_effector_pose(joint_angles)

            # Update visualization
            viz_server.update_robot_state(joint_angles)

            # Print state every second
            if int(t) != int(t - dt):
                print_robot_state(joint_angles, ee_pos, ee_rot)

            t += dt
            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n\n⏹️  Test mode stopped")
        viz_server.stop()


def real_mode(robot_id: str, port: str, update_rate: float = 10.0):
    """Run in real mode, reading from SO-100 hardware.

    Args:
        robot_id: Robot ID matching calibration file
        port: Serial port (e.g., /dev/ttyACM0)
        update_rate: Update frequency in Hz
    """
    print(f"\n🤖 Running in REAL MODE")
    print(f"   Robot ID: {robot_id}")
    print(f"   Serial Port: {port}")
    print(f"   Update Rate: {update_rate} Hz\n")

    # Initialize components
    fk = SO100ForwardKinematics()
    robot = DirectSO100(robot_id=robot_id, port=port)
    viz_server = RobotVizServer(port=5000)
    viz_server.set_mode("real")

    # Connect to robot
    print("Connecting to SO-100...")
    if not robot.connect():
        print("\n❌ Failed to connect to SO-100!")
        print("\nTroubleshooting:")
        print(f"  1. Check that the robot is powered on")
        print(f"  2. Check serial port: {port}")
        print(f"  3. Check calibration file exists:")
        print(f"     ~/.cache/lerobot/calibration/robots/so100_follower/{robot_id}.json")
        print(f"  4. Check permissions: sudo chmod 666 {port}")
        return

    print("✓ Connected to SO-100")

    # Start visualization server
    viz_server.start(blocking=False)
    print(f"\n🌐 Visualization: http://localhost:5000")
    print("\nReading joint positions... Press Ctrl+C to stop\n")

    dt = 1.0 / update_rate
    last_print_time = 0.0

    try:
        while True:
            loop_start = time.time()

            # Read joint positions from robot
            joint_angles = robot.get_joint_positions()

            # Compute forward kinematics
            ee_pos, ee_rot = fk.compute_end_effector_pose(joint_angles)

            # Update visualization
            viz_server.update_robot_state(joint_angles)

            # Print state every second
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                print_robot_state(joint_angles, ee_pos, ee_rot)
                last_print_time = current_time

            # Maintain update rate
            loop_duration = time.time() - loop_start
            sleep_time = max(0, dt - loop_duration)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped")
    finally:
        robot.disconnect()
        viz_server.stop()
        print("✓ Disconnected from SO-100")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SO-100 Leader Arm Forward Kinematics with Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test mode with simulated motion
  python so100_leader_fk.py --test

  # Real mode reading from hardware
  python so100_leader_fk.py --robot-id so100_leader --port /dev/ttyACM0

  # Custom update rate
  python so100_leader_fk.py --robot-id so100_leader --port /dev/ttyACM0 --rate 20
        """
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with simulated joint angles (no hardware required)"
    )

    parser.add_argument(
        "--robot-id",
        type=str,
        help="Robot ID matching calibration file (e.g., 'so100_leader')"
    )

    parser.add_argument(
        "--port",
        type=str,
        help="Serial port for SO-100 (e.g., /dev/ttyACM0)"
    )

    parser.add_argument(
        "--rate",
        type=float,
        default=10.0,
        help="Update rate in Hz (default: 10)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.test and (not args.robot_id or not args.port):
        parser.error("--robot-id and --port are required unless --test is specified")

    # Run appropriate mode
    if args.test:
        test_mode(update_rate=args.rate)
    else:
        real_mode(robot_id=args.robot_id, port=args.port, update_rate=args.rate)


if __name__ == "__main__":
    main()
