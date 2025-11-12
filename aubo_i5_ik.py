#!/usr/bin/env python3
"""AUBO i5 Inverse Kinematics Solver.

This module provides numerical inverse kinematics for the AUBO i5 6-DOF
collaborative robot arm using Jacobian-based iterative methods.

The IK solver uses:
- Damped Least Squares (DLS) / Levenberg-Marquardt method
- Joint limit enforcement
- Orientation and position matching
"""

import numpy as np
from typing import Optional, Tuple
from aubo_i5_kinematics import AuboI5ForwardKinematics


class AuboI5InverseKinematics:
    """Numerical inverse kinematics solver for AUBO i5 robot.

    Uses iterative Jacobian-based method (Damped Least Squares) to solve
    for joint angles that achieve a target end effector pose.
    """

    def __init__(self, fk: Optional[AuboI5ForwardKinematics] = None):
        """Initialize IK solver.

        Args:
            fk: Forward kinematics instance (creates new one if not provided)
        """
        self.fk = fk if fk is not None else AuboI5ForwardKinematics()

    def compute_jacobian(self, joint_angles: np.ndarray, delta: float = 1e-6) -> np.ndarray:
        """Compute numerical Jacobian matrix for current joint configuration.

        The Jacobian relates joint velocities to end effector velocities:
        ẋ = J(q) * q̇

        Args:
            joint_angles: Current 6D joint angles in radians
            delta: Small perturbation for numerical differentiation

        Returns:
            6x6 Jacobian matrix (3 position + 3 orientation derivatives)
        """
        jacobian = np.zeros((6, 6))

        # Get current end effector pose
        ee_pos_0, ee_rot_0 = self.fk.compute_end_effector_pose(joint_angles)

        # Numerical differentiation for each joint
        for i in range(6):
            # Perturb joint i
            joint_angles_perturbed = joint_angles.copy()
            joint_angles_perturbed[i] += delta

            # Compute perturbed end effector pose
            ee_pos_1, ee_rot_1 = self.fk.compute_end_effector_pose(joint_angles_perturbed)

            # Position derivative (∂p/∂qi)
            jacobian[0:3, i] = (ee_pos_1 - ee_pos_0) / delta

            # Orientation derivative (approximated with rotation vector difference)
            # Convert rotation matrices to rotation vectors
            rot_diff = ee_rot_1 @ ee_rot_0.T
            angle = np.arccos(np.clip((np.trace(rot_diff) - 1) / 2, -1.0, 1.0))

            if angle > 1e-6:
                axis = np.array([
                    rot_diff[2, 1] - rot_diff[1, 2],
                    rot_diff[0, 2] - rot_diff[2, 0],
                    rot_diff[1, 0] - rot_diff[0, 1]
                ]) / (2 * np.sin(angle))
                rot_vec_diff = angle * axis
            else:
                rot_vec_diff = np.zeros(3)

            jacobian[3:6, i] = rot_vec_diff / delta

        return jacobian

    def rotation_matrix_to_rotation_vector(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to axis-angle rotation vector.

        Args:
            R: 3x3 rotation matrix

        Returns:
            3D rotation vector (axis * angle)
        """
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))

        if angle < 1e-6:
            return np.zeros(3)

        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) / (2 * np.sin(angle))

        return angle * axis

    def solve(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        max_iterations: int = 100,
        position_tolerance: float = 1e-4,
        orientation_tolerance: float = 1e-3,
        damping: float = 0.1,
        position_weight: float = 1.0,
        orientation_weight: float = 0.5,
        verbose: bool = False
    ) -> Tuple[np.ndarray, bool, int, float]:
        """Solve inverse kinematics for target pose.

        Uses Damped Least Squares (Levenberg-Marquardt) method:
        Δq = J^T(JJ^T + λ²I)^(-1) * e

        Args:
            target_pos: Target position [x, y, z] in meters
            target_rot: Target rotation matrix [3x3]
            initial_guess: Initial joint angles (uses zeros if not provided)
            max_iterations: Maximum iteration count
            position_tolerance: Position error tolerance in meters
            orientation_tolerance: Orientation error tolerance in radians
            damping: Damping factor λ for numerical stability
            position_weight: Weight for position error (0-1)
            orientation_weight: Weight for orientation error (0-1)
            verbose: Print iteration details

        Returns:
            Tuple of:
                - joint_angles: Solution joint angles (6D array)
                - converged: True if solution found within tolerance
                - iterations: Number of iterations used
                - final_error: Final pose error magnitude
        """
        # Initialize joint angles
        if initial_guess is None:
            q = np.zeros(6)
        else:
            q = initial_guess.copy()

        # Ensure within joint limits
        q = self.fk.clip_to_joint_limits(q)

        if verbose:
            print(f"\n{'='*70}")
            print("AUBO i5 Inverse Kinematics Solver")
            print(f"{'='*70}")
            print(f"Target position: {target_pos}")
            print(f"Initial guess: {q}")
            print(f"{'='*70}\n")

        for iteration in range(max_iterations):
            # Compute current end effector pose
            current_pos, current_rot = self.fk.compute_end_effector_pose(q)

            # Compute position error
            pos_error = target_pos - current_pos

            # Compute orientation error (rotation vector from current to target)
            rot_error_matrix = target_rot @ current_rot.T
            rot_error = self.rotation_matrix_to_rotation_vector(rot_error_matrix)

            # Combined error vector (6D: 3 position + 3 orientation)
            error = np.hstack([position_weight * pos_error, orientation_weight * rot_error])

            # Check convergence
            pos_error_mag = np.linalg.norm(pos_error)
            rot_error_mag = np.linalg.norm(rot_error)

            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration:3d}: "
                      f"pos_err={pos_error_mag:.6f} m, "
                      f"rot_err={rot_error_mag:.6f} rad")

            if pos_error_mag < position_tolerance and rot_error_mag < orientation_tolerance:
                if verbose:
                    print(f"\n✓ Converged in {iteration} iterations")
                    print(f"  Final position error: {pos_error_mag:.6f} m")
                    print(f"  Final orientation error: {rot_error_mag:.6f} rad")
                    print(f"  Solution: {q}")
                return q, True, iteration, np.linalg.norm(error)

            # Compute Jacobian
            J = self.compute_jacobian(q)

            # Damped Least Squares update
            # Δq = J^T(JJ^T + λ²I)^(-1) * e
            JJT = J @ J.T
            damping_matrix = damping**2 * np.eye(6)
            delta_q = J.T @ np.linalg.solve(JJT + damping_matrix, error)

            # Update joint angles
            q = q + delta_q

            # Enforce joint limits
            q = self.fk.clip_to_joint_limits(q)

        # Did not converge
        current_pos, current_rot = self.fk.compute_end_effector_pose(q)
        pos_error = target_pos - current_pos
        rot_error_matrix = target_rot @ current_rot.T
        rot_error = self.rotation_matrix_to_rotation_vector(rot_error_matrix)
        error = np.hstack([position_weight * pos_error, orientation_weight * rot_error])

        if verbose:
            print(f"\n⚠ Did not converge after {max_iterations} iterations")
            print(f"  Final position error: {np.linalg.norm(pos_error):.6f} m")
            print(f"  Final orientation error: {np.linalg.norm(rot_error):.6f} rad")
            print(f"  Best solution: {q}")

        return q, False, max_iterations, np.linalg.norm(error)

    def solve_position_only(
        self,
        target_pos: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
        damping: float = 0.1,
        verbose: bool = False
    ) -> Tuple[np.ndarray, bool, int, float]:
        """Solve IK for target position only (ignoring orientation).

        Args:
            target_pos: Target position [x, y, z] in meters
            initial_guess: Initial joint angles
            max_iterations: Maximum iteration count
            tolerance: Position error tolerance in meters
            damping: Damping factor
            verbose: Print iteration details

        Returns:
            Tuple of (joint_angles, converged, iterations, final_error)
        """
        # Initialize joint angles
        if initial_guess is None:
            q = np.zeros(6)
        else:
            q = initial_guess.copy()

        q = self.fk.clip_to_joint_limits(q)

        if verbose:
            print(f"\n{'='*70}")
            print("AUBO i5 IK Solver (Position Only)")
            print(f"{'='*70}")
            print(f"Target position: {target_pos}")
            print(f"{'='*70}\n")

        for iteration in range(max_iterations):
            # Compute current position
            current_pos, _ = self.fk.compute_end_effector_pose(q)

            # Position error
            error = target_pos - current_pos
            error_mag = np.linalg.norm(error)

            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration:3d}: error={error_mag:.6f} m")

            if error_mag < tolerance:
                if verbose:
                    print(f"\n✓ Converged in {iteration} iterations")
                    print(f"  Final error: {error_mag:.6f} m")
                return q, True, iteration, error_mag

            # Compute Jacobian (only use position part)
            J_full = self.compute_jacobian(q)
            J_pos = J_full[0:3, :]  # Only position rows

            # Damped Least Squares
            JJT = J_pos @ J_pos.T
            damping_matrix = damping**2 * np.eye(3)
            delta_q = J_pos.T @ np.linalg.solve(JJT + damping_matrix, error)

            # Update
            q = q + delta_q
            q = self.fk.clip_to_joint_limits(q)

        # Did not converge
        current_pos, _ = self.fk.compute_end_effector_pose(q)
        error = target_pos - current_pos
        error_mag = np.linalg.norm(error)

        if verbose:
            print(f"\n⚠ Did not converge after {max_iterations} iterations")
            print(f"  Final error: {error_mag:.6f} m")

        return q, False, max_iterations, error_mag


def test_aubo_i5_ik():
    """Test AUBO i5 inverse kinematics solver."""
    fk = AuboI5ForwardKinematics()
    ik = AuboI5InverseKinematics(fk)

    print("=" * 70)
    print("AUBO i5 Inverse Kinematics Test")
    print("=" * 70)

    # Test 1: Simple position - arm extended forward
    print("\n" + "="*70)
    print("Test 1: Target position [0.5, 0.0, 0.3] (forward reach)")
    print("="*70)

    target_pos_1 = np.array([0.5, 0.0, 0.3])
    target_rot_1 = np.eye(3)  # Identity orientation

    solution_1, converged_1, iters_1, error_1 = ik.solve(
        target_pos_1, target_rot_1,
        verbose=True
    )

    # Verify with FK
    verify_pos_1, verify_rot_1 = fk.compute_end_effector_pose(solution_1)
    print(f"\nVerification:")
    print(f"  Computed position: {verify_pos_1}")
    print(f"  Target position:   {target_pos_1}")
    print(f"  Position error:    {np.linalg.norm(verify_pos_1 - target_pos_1):.6f} m")

    # Test 2: Position-only IK
    print("\n" + "="*70)
    print("Test 2: Position-only IK for [0.4, 0.2, 0.5]")
    print("="*70)

    target_pos_2 = np.array([0.4, 0.2, 0.5])
    solution_2, converged_2, iters_2, error_2 = ik.solve_position_only(
        target_pos_2,
        verbose=True
    )

    verify_pos_2, _ = fk.compute_end_effector_pose(solution_2)
    print(f"\nVerification:")
    print(f"  Computed position: {verify_pos_2}")
    print(f"  Target position:   {target_pos_2}")
    print(f"  Position error:    {np.linalg.norm(verify_pos_2 - target_pos_2):.6f} m")

    # Test 3: Round-trip test (FK -> IK -> FK)
    print("\n" + "="*70)
    print("Test 3: Round-trip test (FK -> IK -> FK)")
    print("="*70)

    # Generate random joint angles
    test_angles = np.random.uniform(-1.0, 1.0, 6)
    print(f"Original joint angles: {test_angles}")

    # Forward kinematics
    fk_pos, fk_rot = fk.compute_end_effector_pose(test_angles)
    print(f"FK position: {fk_pos}")

    # Inverse kinematics
    ik_solution, converged, iters, error = ik.solve(
        fk_pos, fk_rot,
        initial_guess=np.zeros(6),
        verbose=False
    )
    print(f"IK solution: {ik_solution}")
    print(f"Converged: {converged}, Iterations: {iters}, Error: {error:.6f}")

    # Verify
    verify_pos, verify_rot = fk.compute_end_effector_pose(ik_solution)
    print(f"Verification position: {verify_pos}")
    print(f"Position error: {np.linalg.norm(verify_pos - fk_pos):.6f} m")

    print("\n" + "="*70)
    print("All IK tests completed!")
    print("="*70)


if __name__ == "__main__":
    test_aubo_i5_ik()
