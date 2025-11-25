"""Flask web server for dual-arm robot visualization (SO-100 + AUBO i5)."""

import json
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

try:
    from flask import Flask, render_template, jsonify, send_from_directory
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None
    print("Warning: Flask not installed. Install with: pip install flask flask-cors")

from external.forward_kinematics import SO100ForwardKinematics
from aubo_i5_kinematics import AuboI5ForwardKinematics


class DualArmVizServer:
    """Flask web server for visualizing two robot arms side-by-side.

    Visualizes:
    - SO-100 (5-DOF leader) on the left
    - AUBO i5 (6-DOF follower) on the right

    Usage:
        >>> server = DualArmVizServer(port=5000)
        >>> server.start(blocking=False)
        >>> # Update both robots
        >>> server.update_both(so100_joints, aubo_joints)
        >>> # Or update individually
        >>> server.update_so100(so100_joints)
        >>> server.update_aubo(aubo_joints)
    """

    def __init__(self, port: int = 5000, host: str = "0.0.0.0"):
        """Initialize dual-arm visualization server.

        Args:
            port: Port to run Flask server on
            host: Host address (0.0.0.0 for external access)
        """
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required. Install with: pip install flask flask-cors")

        self.port = port
        self.host = host

        # Flask app setup
        self.app = Flask(
            __name__,
            static_folder=str(Path(__file__).parent / "external" / "static"),
            template_folder=str(Path(__file__).parent / "templates")
        )
        CORS(self.app)  # Enable CORS for API access

        # Forward kinematics for both robots
        self.so100_fk = SO100ForwardKinematics()
        self.aubo_fk = AuboI5ForwardKinematics()

        # System state
        self.mode = "idle"  # idle, test, real
        self.update_count = 0
        self.ik_success_count = 0
        self.ik_total_count = 0

        # SO-100 state (5 joints + gripper)
        self.so100_state = {
            'joint_angles': np.zeros(6).tolist(),
            'joint_positions': [],
            'end_effector_pos': [0.0, 0.0, 0.0],
            'gripper_state': 0.0,
            'timestamp': time.time()
        }

        # AUBO i5 state (6 joints)
        self.aubo_state = {
            'joint_angles': np.zeros(6).tolist(),
            'joint_positions': [],
            'end_effector_pos': [0.0, 0.0, 0.0],
            'end_effector_rot': np.eye(3).tolist(),
            'timestamp': time.time()
        }

        # Thread management
        self._server_thread: Optional[threading.Thread] = None
        self._running = False

        # Setup Flask routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask API routes."""

        @self.app.route('/')
        def index():
            """Serve main dual visualization page."""
            return render_template('dual_robot_viz.html')

        @self.app.route('/api/so100_state')
        def get_so100_state():
            """Get current SO-100 robot state."""
            return jsonify(self.so100_state)

        @self.app.route('/api/aubo_state')
        def get_aubo_state():
            """Get current AUBO i5 robot state."""
            return jsonify(self.aubo_state)

        @self.app.route('/api/system_status')
        def get_system_status():
            """Get overall system status."""
            ik_success_rate = (self.ik_success_count / self.ik_total_count * 100
                              if self.ik_total_count > 0 else 0)

            return jsonify({
                'mode': self.mode,
                'update_count': self.update_count,
                'ik_success_rate': round(ik_success_rate, 1),
                'ik_successes': self.ik_success_count,
                'ik_total': self.ik_total_count,
                'timestamp': time.time()
            })

        @self.app.route('/api/robot_configs')
        def get_robot_configs():
            """Get configuration for both robots."""
            return jsonify({
                'so100': {
                    'name': 'SO-100',
                    'type': 'leader',
                    'dof': 5,
                    'max_reach': 450,  # mm
                    'joint_names': [
                        'Shoulder_Rotation',
                        'Shoulder_Pitch',
                        'Elbow',
                        'Wrist_Pitch',
                        'Wrist_Roll',
                        'Gripper'
                    ]
                },
                'aubo': {
                    'name': 'AUBO i5',
                    'type': 'follower',
                    'dof': 6,
                    'max_reach': 886.5,  # mm
                    'joint_names': [
                        'shoulder_joint',
                        'upperArm_joint',
                        'foreArm_joint',
                        'wrist1_joint',
                        'wrist2_joint',
                        'wrist3_joint'
                    ]
                },
                'workspace_scale': 1.97
            })

    def update_so100(self, joint_angles: np.ndarray):
        """Update SO-100 robot state.

        Args:
            joint_angles: 6D array [5 joint angles + gripper]
        """
        if len(joint_angles) != 6:
            raise ValueError(f"SO-100 expects 6 values, got {len(joint_angles)}")

        # Compute forward kinematics
        fk_result = self.so100_fk.compute_with_gripper(joint_angles)

        # Update state
        self.so100_state = {
            'joint_angles': joint_angles.tolist(),
            'joint_positions': [pos.tolist() for pos in fk_result['joint_positions']],
            'joint_frames': [
                {
                    'position': frame['position'].tolist(),
                    'rotation': frame['rotation'].tolist(),
                    'z_axis': frame['z_axis'].tolist()
                }
                for frame in fk_result['joint_frames']
            ],
            'end_effector_pos': fk_result['end_effector_pos'].tolist(),
            'gripper_state': float(fk_result['gripper_state']),
            'timestamp': time.time()
        }

    def update_aubo(self, joint_angles: np.ndarray):
        """Update AUBO i5 robot state.

        Args:
            joint_angles: 6D array of joint angles (radians)
        """
        if len(joint_angles) != 6:
            raise ValueError(f"AUBO i5 expects 6 values, got {len(joint_angles)}")

        # Compute forward kinematics with full state
        fk_result = self.aubo_fk.compute_with_full_state(joint_angles)

        # Update state
        self.aubo_state = {
            'joint_angles': joint_angles.tolist(),
            'joint_positions': [pos.tolist() for pos in fk_result['joint_positions']],
            'joint_frames': [
                {
                    'position': frame['position'].tolist(),
                    'rotation': frame['rotation'].tolist(),
                    'z_axis': frame['z_axis'].tolist()
                }
                for frame in fk_result['joint_frames']
            ],
            'end_effector_pos': fk_result['end_effector_pos'].tolist(),
            'end_effector_rot': fk_result['end_effector_rot'].tolist(),
            'timestamp': time.time()
        }

    def update_both(self, so100_joints: np.ndarray, aubo_joints: np.ndarray,
                   ik_success: bool = True):
        """Update both robots simultaneously.

        Args:
            so100_joints: SO-100 joint angles (6D)
            aubo_joints: AUBO i5 joint angles (6D)
            ik_success: Whether IK solver converged for AUBO
        """
        self.update_so100(so100_joints)
        self.update_aubo(aubo_joints)

        # Update counters
        self.update_count += 1
        self.ik_total_count += 1
        if ik_success:
            self.ik_success_count += 1

    def set_mode(self, mode: str):
        """Set system mode (idle, test, real).

        Args:
            mode: Mode string ('idle', 'test', 'real')
        """
        self.mode = mode

    def reset_stats(self):
        """Reset statistics counters."""
        self.update_count = 0
        self.ik_success_count = 0
        self.ik_total_count = 0

    def start(self, blocking: bool = False):
        """Start Flask server.

        Args:
            blocking: If True, runs in main thread (blocks). If False, runs in background thread.
        """
        if self._running:
            print(f"Server already running on http://{self.host}:{self.port}")
            return

        print("=" * 70)
        print("🤖 Dual-Arm Robot Visualization Server Starting")
        print("=" * 70)
        print(f"Web Interface: http://localhost:{self.port}")
        print(f"")
        print(f"Robots:")
        print(f"  • SO-100 (Leader)    - 5 DOF, 450mm reach")
        print(f"  • AUBO i5 (Follower) - 6 DOF, 886.5mm reach")
        print(f"")
        print(f"API Endpoints:")
        print(f"  - GET  /api/so100_state     - SO-100 state")
        print(f"  - GET  /api/aubo_state      - AUBO i5 state")
        print(f"  - GET  /api/system_status   - System status")
        print(f"  - GET  /api/robot_configs   - Robot configurations")
        print("=" * 70)

        self._running = True

        if blocking:
            # Run in main thread
            self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)
        else:
            # Run in background thread
            self._server_thread = threading.Thread(
                target=lambda: self.app.run(
                    host=self.host,
                    port=self.port,
                    debug=False,
                    use_reloader=False
                ),
                daemon=True
            )
            self._server_thread.start()
            time.sleep(1)  # Wait for server to start
            print("✓ Server started in background thread")

    def stop(self):
        """Stop Flask server."""
        self._running = False
        print("Server stopped")

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running


def main():
    """Demo: Start dual visualization server with animated motion."""
    import argparse

    parser = argparse.ArgumentParser(description="Dual-Arm Robot Visualization Server")
    parser.add_argument("--port", type=int, default=5000, help="Port to run server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--duration", type=int, default=0,
                       help="Demo duration in seconds (0 = infinite)")
    args = parser.parse_args()

    # Start server
    server = DualArmVizServer(port=args.port, host=args.host)
    server.set_mode("demo")
    server.start(blocking=False)

    # Demo: Animate both robots with coordinated motion
    print("\n🎬 Running demo animation...")
    print("   Watch the visualization at http://localhost:5000")
    print("   Press Ctrl+C to stop\n")

    try:
        start_time = time.time()
        t = 0

        while True:
            # Check duration
            if args.duration > 0 and (time.time() - start_time) > args.duration:
                break

            # SO-100: Sinusoidal motion
            so100_joints = np.array([
                0.3 * np.sin(t),              # Shoulder rotation
                0.4 * np.sin(t * 0.8),        # Shoulder pitch
                0.5 * np.cos(t * 0.6),        # Elbow
                0.3 * np.sin(t * 1.2),        # Wrist pitch
                0.2 * np.cos(t * 1.5),        # Wrist roll
                0.5 + 0.5 * np.sin(t * 0.5)   # Gripper (oscillate 0-1)
            ])

            # AUBO i5: Coordinated motion (slightly different phase)
            aubo_joints = np.array([
                0.4 * np.sin(t + 0.1),
                0.5 * np.sin(t * 0.7 + 0.2),
                0.6 * np.cos(t * 0.5 + 0.1),
                0.3 * np.sin(t * 1.1),
                0.4 * np.cos(t * 1.3),
                0.2 * np.sin(t * 1.4)
            ])

            # Update both robots
            server.update_both(so100_joints, aubo_joints, ik_success=True)

            # Print status every 2 seconds
            if int(t * 10) % 20 == 0:
                elapsed = time.time() - start_time
                print(f"  Updates: {server.update_count} | "
                      f"Time: {elapsed:.1f}s | "
                      f"Rate: {server.update_count/elapsed:.1f} Hz")

            t += 0.1
            time.sleep(0.1)  # 10 Hz update

        print("\n✓ Demo completed successfully")
        print(f"  Total updates: {server.update_count}")
        print(f"  Duration: {time.time() - start_time:.1f}s")

    except KeyboardInterrupt:
        print("\n\n⏹️  Demo stopped by user")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
