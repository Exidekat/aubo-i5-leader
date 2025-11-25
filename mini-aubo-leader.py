#!/usr/bin/env python3
"""Mini-Aubo Leader: Direct leader-follower control.

Controls a real AUBO i5 robot using a Mini-Aubo leader (Feetech replica).
Uses delta positions from home to achieve synchronized motion.

The Mini-Aubo is a 6-DOF replica of the AUBO i5 built with Feetech motors.
Each motor ID 1-6 corresponds directly to AUBO i5 joints 1-6.

Usage:
    python mini-aubo-leader.py \\
        --leader-port /dev/ttyACM0 \\
        --aubo-ip 192.168.1.100 \\
        --rate 10 \\
        --verbose
"""

import sys
import time
import signal
import argparse
import numpy as np
import serial
from typing import Optional, Tuple

try:
    import pyaubo
    PYAUBO_AVAILABLE = True
except ImportError:
    PYAUBO_AVAILABLE = False
    print("Warning: pyaubo-sdk not installed. Install with: pip install pyaubo-sdk")


class FeetechReader:
    """Read joint positions from Feetech motors (Mini-Aubo Leader).

    Implements the Feetech serial protocol to read raw motor positions
    without calibration. Returns raw encoder values (0-4095 range).
    """

    def __init__(self, port: str, baudrate: int = 1000000, num_motors: int = 6):
        """Initialize Feetech reader.

        Args:
            port: Serial port (e.g., '/dev/ttyACM0')
            baudrate: Communication speed (default: 1000000)
            num_motors: Number of motors to read (default: 6)
        """
        self.port = port
        self.baudrate = baudrate
        self.num_motors = num_motors
        self.serial_conn: Optional[serial.Serial] = None

    def connect(self) -> bool:
        """Open serial connection to Feetech motors.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1
            )
            time.sleep(0.5)  # Wait for connection to stabilize

            # Ping motor 1 to verify connection
            if self._ping_motor(1):
                print(f"✓ Connected to Mini-Aubo on {self.port}")
                return True
            else:
                print(f"✗ Failed to ping motor 1 on {self.port}")
                return False

        except serial.SerialException as e:
            print(f"✗ Serial connection failed: {e}")
            return False

    def _ping_motor(self, motor_id: int) -> bool:
        """Ping a motor to check if it's responding.

        Args:
            motor_id: Motor ID (1-6)

        Returns:
            True if motor responds, False otherwise
        """
        if not self.serial_conn:
            return False

        # Feetech ping command: [0xFF, 0xFF, ID, LENGTH, INSTRUCTION, CHECKSUM]
        packet = [0xFF, 0xFF, motor_id, 0x02, 0x01]  # 0x01 = PING
        checksum = (~sum(packet[2:])) & 0xFF
        packet.append(checksum)

        try:
            self.serial_conn.write(bytes(packet))
            response = self.serial_conn.read(6)
            return len(response) == 6 and response[0] == 0xFF and response[1] == 0xFF
        except:
            return False

    def _read_motor_position(self, motor_id: int) -> Optional[int]:
        """Read raw position from a single motor.

        Args:
            motor_id: Motor ID (1-6)

        Returns:
            Raw position (0-4095) or None if read fails
        """
        if not self.serial_conn:
            return None

        # Feetech read command
        # [0xFF, 0xFF, ID, LENGTH, INSTRUCTION, ADDRESS, DATA_LENGTH, CHECKSUM]
        packet = [0xFF, 0xFF, motor_id, 0x04, 0x02, 0x38, 0x02]  # 0x02 = READ, 0x38 = position register
        checksum = (~sum(packet[2:])) & 0xFF
        packet.append(checksum)

        try:
            # Clear buffer
            self.serial_conn.reset_input_buffer()

            # Send read command
            self.serial_conn.write(bytes(packet))

            # Read response: [0xFF, 0xFF, ID, LENGTH, ERROR, DATA_LOW, DATA_HIGH, CHECKSUM]
            response = self.serial_conn.read(8)

            if len(response) == 8 and response[0] == 0xFF and response[1] == 0xFF:
                # Position is 14-bit value (0-4095): combine low and high bytes
                position = (response[6] << 8) | response[5]
                return position & 0x3FFF  # Mask to 14 bits
            else:
                return None

        except Exception as e:
            print(f"Warning: Failed to read motor {motor_id}: {e}")
            return None

    def read_all_motors(self) -> np.ndarray:
        """Read raw positions from all motors.

        Returns:
            Array of 6 raw positions (0-4095). Returns 0 for failed reads.
        """
        positions = np.zeros(self.num_motors, dtype=np.int32)

        for motor_id in range(1, self.num_motors + 1):
            pos = self._read_motor_position(motor_id)
            if pos is not None:
                positions[motor_id - 1] = pos
            else:
                print(f"Warning: Motor {motor_id} read failed, using 0")
                positions[motor_id - 1] = 0

        return positions

    def close(self):
        """Close serial connection."""
        if self.serial_conn:
            self.serial_conn.close()
            print("✓ Mini-Aubo connection closed")


class AuboController:
    """Control AUBO i5 robot using pyaubo-sdk."""

    # AUBO i5 joint limits: ±175° = ±3.054 radians
    JOINT_LIMIT_RAD = 3.054

    def __init__(self, ip_address: str):
        """Initialize AUBO controller.

        Args:
            ip_address: IP address of AUBO i5 robot
        """
        if not PYAUBO_AVAILABLE:
            raise ImportError("pyaubo-sdk not installed. Install with: pip install pyaubo-sdk")

        self.ip_address = ip_address
        self.robot = None
        self.connected = False

    def connect(self) -> bool:
        """Connect to AUBO i5 robot.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Initialize AUBO SDK
            self.robot = pyaubo.Robot(self.ip_address)

            # Attempt connection
            result = self.robot.connect()
            if result:
                print(f"✓ Connected to AUBO i5 at {self.ip_address}")
                self.connected = True

                # Enable robot (if needed)
                self.robot.enable_robot()
                return True
            else:
                print(f"✗ Failed to connect to AUBO i5 at {self.ip_address}")
                return False

        except Exception as e:
            print(f"✗ AUBO connection error: {e}")
            return False

    def read_joint_positions(self) -> np.ndarray:
        """Read current joint positions from AUBO i5.

        Returns:
            Array of 6 joint angles in radians
        """
        if not self.connected or not self.robot:
            return np.zeros(6)

        try:
            # Get current joint positions
            positions = self.robot.get_joint_positions()
            return np.array(positions[:6])  # Ensure 6 DOF

        except Exception as e:
            print(f"Warning: Failed to read AUBO positions: {e}")
            return np.zeros(6)

    def move_to_joints(self, joint_angles: np.ndarray, blocking: bool = False) -> bool:
        """Send target joint positions to AUBO i5.

        Args:
            joint_angles: Array of 6 joint angles in radians
            blocking: If True, waits for motion to complete

        Returns:
            True if command sent successfully, False otherwise
        """
        if not self.connected or not self.robot:
            return False

        # Clip to joint limits for safety
        clipped = np.clip(joint_angles, -self.JOINT_LIMIT_RAD, self.JOINT_LIMIT_RAD)

        try:
            # Send move command
            self.robot.move_to_joint_positions(clipped.tolist(), blocking=blocking)
            return True

        except Exception as e:
            print(f"Warning: Failed to send AUBO command: {e}")
            return False

    def disconnect(self):
        """Disconnect from AUBO i5."""
        if self.robot:
            try:
                self.robot.disable_robot()
                self.robot.disconnect()
                print("✓ AUBO i5 disconnected")
            except:
                pass
        self.connected = False


class MiniAuboLeader:
    """Main controller for Mini-Aubo leader -> AUBO i5 follower system.

    Reads Mini-Aubo (Feetech) positions, calculates deltas from home,
    and applies those deltas to AUBO i5 to create synchronized motion.
    """

    # Feetech motors: 14-bit resolution (0-4095) = 360 degrees
    FEETECH_RESOLUTION = 4096
    FEETECH_TO_RAD = (2 * np.pi) / FEETECH_RESOLUTION

    def __init__(self, leader_port: str, aubo_ip: str, verbose: bool = False):
        """Initialize Mini-Aubo Leader controller.

        Args:
            leader_port: Serial port for Mini-Aubo (Feetech motors)
            aubo_ip: IP address of AUBO i5 robot
            verbose: Enable verbose status printing
        """
        self.verbose = verbose
        self.running = False

        # Initialize hardware interfaces
        self.leader = FeetechReader(port=leader_port, num_motors=6)
        self.aubo = AuboController(ip_address=aubo_ip)

        # Home positions (captured at startup)
        self.leader_home: Optional[np.ndarray] = None  # Raw positions (0-4095)
        self.aubo_home: Optional[np.ndarray] = None    # Joint angles (radians)

        # Statistics
        self.update_count = 0
        self.start_time = 0

    def initialize(self) -> bool:
        """Initialize connections and capture home positions.

        Returns:
            True if initialization successful, False otherwise
        """
        print("=" * 70)
        print("🤖 Mini-Aubo Leader Initialization")
        print("=" * 70)

        # Connect to Mini-Aubo (leader)
        if not self.leader.connect():
            print("\n✗ Failed to connect to Mini-Aubo")
            return False

        # Connect to AUBO i5 (follower)
        if not self.aubo.connect():
            print("\n✗ Failed to connect to AUBO i5")
            self.leader.close()
            return False

        print("\n📍 Capturing home positions...")
        print("   (Ensure both robots are in desired home configuration)")
        time.sleep(1)

        # Read Mini-Aubo home position (raw)
        self.leader_home = self.leader.read_all_motors()
        print(f"\n✓ Mini-Aubo home (raw): {self.leader_home}")

        # Read AUBO i5 home position (radians)
        self.aubo_home = self.aubo.read_joint_positions()
        print(f"✓ AUBO i5 home (rad):   {np.round(self.aubo_home, 3)}")
        print(f"✓ AUBO i5 home (deg):   {np.round(np.degrees(self.aubo_home), 1)}")

        # Verify valid readings
        if np.all(self.leader_home == 0):
            print("\n⚠️  Warning: All Mini-Aubo positions are 0 - check connection")
            return False

        print("\n" + "=" * 70)
        print("✓ Initialization complete! Ready to start control loop.")
        print("=" * 70)
        return True

    def run(self, update_rate: float = 10.0):
        """Run main control loop.

        Args:
            update_rate: Update rate in Hz (default: 10)
        """
        if self.leader_home is None or self.aubo_home is None:
            print("✗ Not initialized. Call initialize() first.")
            return

        dt = 1.0 / update_rate
        self.running = True
        self.start_time = time.time()

        print(f"\n🎬 Starting control loop at {update_rate} Hz")
        print("   Press Ctrl+C to stop\n")

        try:
            while self.running:
                loop_start = time.time()

                # 1. Read current Mini-Aubo positions (raw)
                leader_current = self.leader.read_all_motors()

                # 2. Calculate delta from home (raw counts)
                leader_delta_raw = leader_current - self.leader_home

                # 3. Convert delta to radians
                leader_delta_rad = leader_delta_raw * self.FEETECH_TO_RAD

                # 4. Calculate AUBO target positions
                aubo_target = self.aubo_home + leader_delta_rad

                # 5. Send to AUBO i5 (with joint limit clipping)
                success = self.aubo.move_to_joints(aubo_target, blocking=False)

                # 6. Update statistics
                self.update_count += 1

                # 7. Print status (if verbose)
                if self.verbose and self.update_count % 10 == 0:
                    elapsed = time.time() - self.start_time
                    rate = self.update_count / elapsed if elapsed > 0 else 0
                    print(f"[{elapsed:6.1f}s] Updates: {self.update_count:4d} | "
                          f"Rate: {rate:5.1f} Hz | "
                          f"Delta (deg): {np.round(np.degrees(leader_delta_rad), 1)}")

                # 8. Sleep to maintain update rate
                loop_time = time.time() - loop_start
                if loop_time < dt:
                    time.sleep(dt - loop_time)

        except KeyboardInterrupt:
            print("\n\n⏹️  Stopped by user")

        finally:
            self.shutdown()

    def shutdown(self):
        """Cleanup and close connections."""
        self.running = False

        # Print final statistics
        if self.update_count > 0:
            elapsed = time.time() - self.start_time
            avg_rate = self.update_count / elapsed if elapsed > 0 else 0

            print("\n" + "=" * 70)
            print("📊 Session Statistics")
            print("=" * 70)
            print(f"Duration:      {elapsed:.1f}s")
            print(f"Total updates: {self.update_count}")
            print(f"Average rate:  {avg_rate:.1f} Hz")
            print("=" * 70)

        # Close connections
        self.leader.close()
        self.aubo.disconnect()

        print("\n✓ Shutdown complete")


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\n⚠️  Interrupt signal received")
    sys.exit(0)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Mini-Aubo Leader: Direct leader-follower control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python mini-aubo-leader.py --leader-port /dev/ttyACM0 --aubo-ip 192.168.1.100

  # With verbose output at 20 Hz
  python mini-aubo-leader.py \\
      --leader-port /dev/ttyACM0 \\
      --aubo-ip 192.168.1.100 \\
      --rate 20 \\
      --verbose

How it works:
  1. On startup, captures home positions of both robots
  2. Continuously reads Mini-Aubo positions (Feetech motors)
  3. Calculates delta from home: delta = current - home
  4. Applies delta to AUBO i5: target = aubo_home + delta
  5. Sends target positions to AUBO i5 in real-time
        """
    )

    parser.add_argument(
        "--leader-port",
        type=str,
        required=True,
        help="Serial port for Mini-Aubo leader (e.g., /dev/ttyACM0)"
    )

    parser.add_argument(
        "--aubo-ip",
        type=str,
        required=True,
        help="IP address of AUBO i5 follower (e.g., 192.168.1.100)"
    )

    parser.add_argument(
        "--rate",
        type=float,
        default=10.0,
        help="Update rate in Hz (default: 10)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose status output"
    )

    args = parser.parse_args()

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    # Check pyaubo availability
    if not PYAUBO_AVAILABLE:
        print("✗ Error: pyaubo-sdk not installed")
        print("  Install with: pip install pyaubo-sdk")
        sys.exit(1)

    # Create controller
    controller = MiniAuboLeader(
        leader_port=args.leader_port,
        aubo_ip=args.aubo_ip,
        verbose=args.verbose
    )

    # Initialize
    if not controller.initialize():
        print("\n✗ Initialization failed")
        sys.exit(1)

    # Run control loop
    controller.run(update_rate=args.rate)


if __name__ == "__main__":
    main()
