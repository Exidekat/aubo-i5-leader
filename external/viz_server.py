"""Flask web server for real-time robot arm visualization."""

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

from forward_kinematics import SO100ForwardKinematics


class RobotVizServer:
    """Flask web server for visualizing robot arm in 3D.

    Serves a Three.js-based 3D visualization showing:
    - Current robot joint positions
    - End effector position
    - Action trajectory predictions from VLA

    Usage:
        >>> server = RobotVizServer(port=5000)
        >>> server.start()
        >>> # Update robot state
        >>> server.update_robot_state(joint_angles)
        >>> # Add predicted actions
        >>> server.add_action_trajectory(predicted_actions)
        >>> # Server runs in background thread
        >>> # Access at http://localhost:5000
    """

    def __init__(self, port: int = 5000, host: str = "0.0.0.0"):
        """Initialize visualization server.

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
            static_folder=str(Path(__file__).parent / "static"),
            template_folder=str(Path(__file__).parent / "templates")
        )
        CORS(self.app)  # Enable CORS for API access

        # Robot state
        self.fk = SO100ForwardKinematics()
        self.mode = "idle"  # Can be: idle, demo, real, simulation
        self.current_state = {
            'joint_angles': np.zeros(6).tolist(),
            'joint_positions': [],
            'end_effector_pos': [0.0, 0.0, 0.0],
            'gripper_state': 0.0,
            'timestamp': time.time(),
            'mode': self.mode
        }

        # Action trajectory (list of predicted states)
        self.action_trajectory: List[Dict[str, Any]] = []
        self.max_trajectory_length = 50  # Keep last 50 predictions

        # Thread management
        self._server_thread: Optional[threading.Thread] = None
        self._running = False

        # Setup Flask routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask API routes."""

        @self.app.route('/')
        def index():
            """Serve main visualization page."""
            return render_template('robot_viz.html')

        @self.app.route('/api/robot_state')
        def get_robot_state():
            """Get current robot state (joint positions in 3D).

            Supports optional query params j0, j1, j2, j3, j4, j5 to compute FK
            for custom joint angles (for testing).
            """
            from flask import request

            # Check if custom joint angles provided via query params
            joint_params = [request.args.get(f'j{i}') for i in range(6)]
            if all(p is not None for p in joint_params):
                # Compute FK for custom angles
                try:
                    test_angles = np.array([float(p) for p in joint_params])
                    fk_result = self.fk.compute_with_gripper(test_angles)

                    test_state = {
                        'joint_angles': fk_result['joint_angles'].tolist(),
                        'joint_positions': [pos.tolist() for pos in fk_result['joint_positions']],
                        'joint_frames': [
                            {
                                'position': f['position'].tolist(),
                                'rotation': f['rotation'].tolist(),
                                'z_axis': f['z_axis'].tolist()
                            }
                            for f in fk_result['joint_frames']
                        ],
                        'end_effector_pos': fk_result['end_effector_pos'].tolist(),
                        'gripper_state': fk_result['gripper_state'],
                        'mode': 'test'
                    }
                    return jsonify(test_state)
                except (ValueError, KeyError) as e:
                    return jsonify({'error': str(e)}), 400

            # Return current state
            return jsonify(self.current_state)

        @self.app.route('/api/action_trajectory')
        def get_action_trajectory():
            """Get predicted action trajectory."""
            return jsonify({
                'trajectory': self.action_trajectory,
                'length': len(self.action_trajectory)
            })

        @self.app.route('/api/robot_config')
        def get_robot_config():
            """Get robot configuration."""
            return jsonify({
                'robot_type': 'SO-100',
                'dof': 6,
                'num_joints': self.fk.num_joints,
                'joint_names': [
                    'shoulder_pan',
                    'shoulder_lift',
                    'elbow_flex',
                    'wrist_flex',
                    'wrist_roll',
                    'gripper'
                ]
            })

    def update_robot_state(self, joint_angles: np.ndarray):
        """Update current robot state and compute forward kinematics.

        Args:
            joint_angles: 6D array [5 joint angles + gripper]
        """
        if len(joint_angles) != 6:
            raise ValueError(f"Expected 6 joint values, got {len(joint_angles)}")

        # Compute forward kinematics
        fk_result = self.fk.compute_with_gripper(joint_angles)

        # Convert numpy arrays to lists for JSON serialization
        self.current_state = {
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
            'timestamp': time.time(),
            'mode': self.mode
        }

    def set_mode(self, mode: str):
        """Set visualization mode (idle, demo, real, simulation).

        Args:
            mode: Mode string ('idle', 'demo', 'real', 'simulation')
        """
        self.mode = mode
        self.current_state['mode'] = mode

    def add_action_trajectory(self, predicted_actions: np.ndarray):
        """Add predicted action sequence to trajectory visualization.

        Args:
            predicted_actions: Array of shape (horizon, 6) with predicted actions
                              Each row is [5 joint angles + gripper]
        """
        if predicted_actions.ndim != 2 or predicted_actions.shape[1] != 6:
            raise ValueError(f"Expected shape (horizon, 6), got {predicted_actions.shape}")

        # Clear old trajectory
        self.action_trajectory = []

        # Compute FK for each predicted action
        for action in predicted_actions:
            fk_result = self.fk.compute_with_gripper(action)
            state = {
                'joint_angles': action.tolist(),
                'joint_positions': [pos.tolist() for pos in fk_result['joint_positions']],
                'end_effector_pos': fk_result['end_effector_pos'].tolist(),
                'gripper_state': float(fk_result['gripper_state']),
            }
            self.action_trajectory.append(state)

        # Limit trajectory length
        if len(self.action_trajectory) > self.max_trajectory_length:
            self.action_trajectory = self.action_trajectory[-self.max_trajectory_length:]

    def clear_trajectory(self):
        """Clear the action trajectory."""
        self.action_trajectory = []

    def start(self, blocking: bool = False):
        """Start Flask server.

        Args:
            blocking: If True, runs in main thread (blocks). If False, runs in background thread.
        """
        if self._running:
            print(f"Server already running on http://{self.host}:{self.port}")
            return

        print("=" * 60)
        print("🤖 Robot Visualization Server Starting")
        print("=" * 60)
        print(f"Web Interface: http://localhost:{self.port}")
        print(f"API Endpoints:")
        print(f"  - GET  /api/robot_state      - Current robot state")
        print(f"  - GET  /api/action_trajectory - Predicted actions")
        print(f"  - GET  /api/robot_config      - Robot configuration")
        print("=" * 60)

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
    """Demo: Start visualization server with sample robot movements."""
    import argparse

    parser = argparse.ArgumentParser(description="Robot Visualization Server")
    parser.add_argument("--port", type=int, default=5000, help="Port to run server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--demo", action="store_true", help="Run demo with animated robot")
    args = parser.parse_args()

    # Start server
    server = RobotVizServer(port=args.port, host=args.host)

    if args.demo:
        # Start server in background
        server.start(blocking=False)

        # Demo: Animate robot through various positions
        print("\n🎬 Running demo animation...")
        print("   Watch the visualization at http://localhost:5000")

        try:
            t = 0
            while True:
                # Sinusoidal joint motion
                joint_angles = np.array([
                    0.3 * np.sin(t),              # Shoulder pan
                    0.4 * np.sin(t * 0.8),        # Shoulder lift
                    0.5 * np.cos(t * 0.6),        # Elbow flex
                    0.3 * np.sin(t * 1.2),        # Wrist flex
                    0.2 * np.cos(t * 1.5),        # Wrist roll
                    0.5 + 0.5 * np.sin(t * 0.5)   # Gripper (oscillate 0-1)
                ])

                server.update_robot_state(joint_angles)

                # Generate predicted trajectory (50 steps ahead)
                horizon = 50
                future_t = np.linspace(t, t + 5, horizon)
                predicted_actions = np.array([
                    [
                        0.3 * np.sin(ft),
                        0.4 * np.sin(ft * 0.8),
                        0.5 * np.cos(ft * 0.6),
                        0.3 * np.sin(ft * 1.2),
                        0.2 * np.cos(ft * 1.5),
                        0.5 + 0.5 * np.sin(ft * 0.5)
                    ]
                    for ft in future_t
                ])
                server.add_action_trajectory(predicted_actions)

                t += 0.05
                time.sleep(0.05)  # 20 Hz update

        except KeyboardInterrupt:
            print("\n\n⏹️  Demo stopped")
            server.stop()

    else:
        # Just start server (blocking)
        server.start(blocking=True)


if __name__ == "__main__":
    main()
