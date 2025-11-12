"""Direct SO-100 control bypassing LeRobot's buggy initialization.

This module provides direct Feetech protocol communication for reading
joint positions from the SO-100 robot without using LeRobot's motor bus.
"""

import json
import time
import serial
import numpy as np
from pathlib import Path
from typing import Dict, Optional


class DirectSO100:
    """Direct SO-100 interface using Feetech protocol.

    This bypasses LeRobot's buggy FeetechMotorsBus initialization
    and reads motor positions directly via serial.
    """

    # SO-100 joint names in order
    JOINT_NAMES = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper"
    ]

    def __init__(self, robot_id: str, port: str, baudrate: int = 1000000):
        """Initialize direct SO-100 connection.

        Args:
            robot_id: Robot ID matching calibration file name
            port: Serial port (e.g., /dev/ttyACM0)
            baudrate: Baud rate (default 1000000 for SO-100)
        """
        self.robot_id = robot_id
        self.port = port
        self.baudrate = baudrate
        self.ser: Optional[serial.Serial] = None
        self.calibration: Optional[Dict] = None

    def connect(self) -> bool:
        """Connect to robot and load calibration.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Load calibration file
            calib_file = (
                Path.home() / ".cache" / "lerobot" / "calibration"
                / "robots" / "so100_follower" / f"{self.robot_id}.json"
            )

            if not calib_file.exists():
                print(f"❌ Calibration file not found: {calib_file}")
                return False

            with open(calib_file, 'r') as f:
                self.calibration = json.load(f)

            # Open serial connection
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )

            # Test connection by pinging motor 1
            time.sleep(0.1)  # Allow port to stabilize
            if not self._ping_motor(1):
                print(f"❌ Failed to ping motor 1 on {self.port}")
                self.ser.close()
                return False

            print(f"✓ Connected to SO-100 '{self.robot_id}' on {self.port}")
            return True

        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from robot."""
        if self.ser and self.ser.is_open:
            self.ser.close()

    def _calculate_checksum(self, data):
        """Calculate Feetech protocol checksum."""
        return (~sum(data)) & 0xFF

    def _ping_motor(self, motor_id: int) -> bool:
        """Ping a motor to check if it's responsive.

        Args:
            motor_id: Motor ID (1-6)

        Returns:
            True if motor responds, False otherwise
        """
        # Feetech ping: [0xFF, 0xFF, ID, Length, Instruction, Checksum]
        packet = [0xFF, 0xFF, motor_id, 0x02, 0x01]
        checksum = self._calculate_checksum(packet[2:])
        packet.append(checksum)

        self.ser.reset_input_buffer()
        self.ser.write(bytes(packet))
        time.sleep(0.01)

        return self.ser.in_waiting > 0

    def _read_motor_position(self, motor_id: int) -> Optional[int]:
        """Read raw position from motor.

        Args:
            motor_id: Motor ID (1-6)

        Returns:
            Raw motor position (0-4095) or None if read failed
        """
        # Feetech read position: Address 0x38 (56), Length 2
        packet = [0xFF, 0xFF, motor_id, 0x04, 0x02, 0x38, 0x02]
        checksum = self._calculate_checksum(packet[2:])
        packet.append(checksum)

        self.ser.reset_input_buffer()
        self.ser.write(bytes(packet))
        time.sleep(0.01)

        if self.ser.in_waiting > 0:
            response = self.ser.read(self.ser.in_waiting)
            if len(response) >= 7:
                pos_low = response[5]
                pos_high = response[6]
                return (pos_high << 8) | pos_low

        return None

    def _raw_to_radians(self, raw_pos: int, joint_name: str) -> float:
        """Convert raw motor position to radians.

        Args:
            raw_pos: Raw position (0-4095)
            joint_name: Joint name from calibration

        Returns:
            Position in radians
        """
        if not self.calibration or joint_name not in self.calibration:
            return 0.0

        calib = self.calibration[joint_name]
        home = calib["homing_offset"]
        range_min = calib["range_min"]
        range_max = calib["range_max"]

        # Calculate position relative to home
        relative = raw_pos - home

        # Convert to radians (-π to π range)
        # Assuming full range is approximately 300 degrees (5.23 radians)
        if relative >= 0:
            max_range = range_max - home
            if max_range > 0:
                normalized = (relative / max_range)
            else:
                normalized = 0.0
        else:
            min_range = home - range_min
            if min_range > 0:
                normalized = (relative / min_range)
            else:
                normalized = 0.0

        # Convert normalized (-1 to 1) to radians (-2.5 to 2.5 approx)
        # Note: This is approximate; actual joint limits vary
        return normalized * 2.5

    def get_joint_positions(self) -> np.ndarray:
        """Read current joint positions from all motors.

        Returns:
            Array of 6 joint angles in radians
        """
        if not self.ser or not self.ser.is_open:
            return np.zeros(6)

        positions = []

        for i, joint_name in enumerate(self.JOINT_NAMES):
            motor_id = i + 1
            raw_pos = self._read_motor_position(motor_id)

            if raw_pos is not None:
                angle = self._raw_to_radians(raw_pos, joint_name)
                positions.append(angle)
            else:
                # If read failed, use last known position or zero
                positions.append(0.0)

        return np.array(positions)

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
