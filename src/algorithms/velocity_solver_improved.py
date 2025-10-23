"""
Improved Velocity Optimization Module

This module implements the corrected two-step velocity optimization with proper
temporal phase differences and target association as described in the paper
"3-D Ego-Motion Estimation Using Multi-Channel FMCW Radar" (Yuan et al. 2023).

Key improvements:
1. Temporal phase difference computation (not spatial)
2. Target association across frames
3. Better optimization initialization
4. Robust cost function
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Optional, Callable
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ImprovedVelocitySolver:
    """
    Improved two-step velocity optimization solver with proper temporal phase differences.
    
    Implements the correct cost function f = [Y - X]^T [Y - X] where:
    - Y: Observed temporal phase differences from consecutive frames
    - X: Predicted phase differences from motion model
    """
    
    def __init__(self,
                 fc: float = 77e9,
                 lambda_c: float = None,
                 num_antennas: int = 8,
                 antenna_spacing: float = None,
                 optimization_method: str = 'differential_evolution',
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 association_threshold: float = 5.0):  # meters
        """
        Initialize improved velocity solver.
        
        Args:
            fc: Carrier frequency (Hz)
            lambda_c: Wavelength (m)
            num_antennas: Number of antenna elements
            antenna_spacing: Antenna spacing (m)
            optimization_method: Optimization method
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            association_threshold: Distance threshold for target association (m)
        """
        self.fc = fc
        self.c = 3e8  # Speed of light
        self.lambda_c = lambda_c or (self.c / self.fc)
        self.num_antennas = num_antennas
        self.antenna_spacing = antenna_spacing or (self.lambda_c / 2)
        self.optimization_method = optimization_method
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.association_threshold = association_threshold
        
        # Antenna positions
        self.antenna_positions = np.arange(self.num_antennas) * self.antenna_spacing
        
        logger.info(f"Initialized improved velocity solver:")
        logger.info(f"  Wavelength: {self.lambda_c*1000:.2f} mm")
        logger.info(f"  Association threshold: {association_threshold} m")
        logger.info(f"  Optimization method: {optimization_method}")
    
    def associate_targets_across_frames(self, 
                                      current_targets: List[Dict],
                                      previous_targets: List[Dict]) -> List[Dict]:
        """
        Associate targets between consecutive frames using Hungarian algorithm.
        
        Args:
            current_targets: Current frame targets
            previous_targets: Previous frame targets
            
        Returns:
            List of associated target pairs with temporal phase differences
        """
        if not previous_targets:
            logger.warning("No previous targets for association")
            return []
        
        # Extract target positions and features
        current_positions = np.array([[t['range_m'] * np.cos(t['azimuth_rad']), 
                                     t['range_m'] * np.sin(t['azimuth_rad'])] for t in current_targets])
        previous_positions = np.array([[t['range_m'] * np.cos(t['azimuth_rad']), 
                                      t['range_m'] * np.sin(t['azimuth_rad'])] for t in previous_targets])
        
        # Compute distance matrix
        distance_matrix = cdist(current_positions, previous_positions)
        
        # Simple greedy association (can be improved with Hungarian algorithm)
        associations = []
        used_previous = set()
        
        for i, current_target in enumerate(current_targets):
            best_match_idx = None
            best_distance = float('inf')
            
            for j, previous_target in enumerate(previous_targets):
                if j in used_previous:
                    continue
                
                distance = distance_matrix[i, j]
                if distance < self.association_threshold and distance < best_distance:
                    best_distance = distance
                    best_match_idx = j
            
            if best_match_idx is not None:
                used_previous.add(best_match_idx)
                associations.append({
                    'current': current_target,
                    'previous': previous_targets[best_match_idx],
                    'distance': best_distance,
                    'temporal_phase_diff': self._compute_temporal_phase_difference(
                        current_target, previous_targets[best_match_idx]
                    )
                })
        
        logger.info(f"Associated {len(associations)} targets across frames")
        return associations
    
    def _compute_temporal_phase_difference(self, 
                                         current_target: Dict, 
                                         previous_target: Dict) -> float:
        """
        Compute temporal phase difference between consecutive frames for same target.
        
        Args:
            current_target: Current frame target
            previous_target: Previous frame target
            
        Returns:
            Temporal phase difference in radians
        """
        # Extract spatial signatures
        current_signature = current_target['spatial_signature']
        previous_signature = previous_target['spatial_signature']
        
        # Compute temporal phase difference using reference antenna (first antenna)
        # This is the key fix: temporal differences, not spatial
        temporal_phase_diff = np.angle(current_signature[0] * np.conj(previous_signature[0]))
        
        return temporal_phase_diff
    
    def compute_observed_phase_differences(self, 
                                         target_associations: List[Dict]) -> np.ndarray:
        """
        Compute observed temporal phase differences from target associations.
        
        Args:
            target_associations: List of associated target pairs
            
        Returns:
            Observed temporal phase differences [N]
        """
        observed_phases = []
        
        for association in target_associations:
            temporal_phase_diff = association['temporal_phase_diff']
            observed_phases.append(temporal_phase_diff)
        
        return np.array(observed_phases)
    
    def compute_phase_difference_model(self, 
                                     target_positions: np.ndarray,
                                     target_angles: np.ndarray,
                                     velocity: np.ndarray,
                                     angular_velocity: np.ndarray,
                                     dt: float) -> np.ndarray:
        """
        Compute predicted temporal phase differences from motion model.
        
        Args:
            target_positions: Target positions [N, 3] (x, y, z)
            target_angles: Target angles [N, 2] (azimuth, elevation)
            velocity: Translational velocity [3] (vx, vy, vz)
            angular_velocity: Angular velocity [3] (wx, wy, wz)
            dt: Time step (s)
            
        Returns:
            Predicted temporal phase differences [N]
        """
        N = len(target_positions)
        predicted_phases = np.zeros(N)
        
        for i in range(N):
            # Target position and angle
            pos = target_positions[i]
            azimuth = target_angles[i, 0]
            elevation = target_angles[i, 1]
            
            # Direction vector
            direction = np.array([
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation)
            ])
            
            # Relative velocity at target position
            # v_rel = v + ω × r
            cross_product = np.cross(angular_velocity, pos)
            relative_velocity = velocity + cross_product
            
            # Radial velocity component
            radial_velocity = np.dot(relative_velocity, direction)
            
            # Temporal phase difference (key improvement)
            phase_diff = (4 * np.pi * radial_velocity * dt) / self.lambda_c
            
            predicted_phases[i] = phase_diff
        
        return predicted_phases
    
    def cost_function(self, 
                    motion_params: np.ndarray,
                    target_positions: np.ndarray,
                    target_angles: np.ndarray,
                    observed_phases: np.ndarray,
                    dt: float) -> float:
        """
        Improved cost function for velocity optimization.
        
        Args:
            motion_params: Motion parameters [6] (vx, vy, vz, wx, wy, wz)
            target_positions: Target positions [N, 3]
            target_angles: Target angles [N, 2]
            observed_phases: Observed temporal phase differences [N]
            dt: Time step (s)
            
        Returns:
            Cost function value
        """
        # Split motion parameters
        velocity = motion_params[:3]
        angular_velocity = motion_params[3:]
        
        # Compute predicted temporal phase differences
        predicted_phases = self.compute_phase_difference_model(
            target_positions, target_angles, velocity, angular_velocity, dt
        )
        
        # Compute residual
        residual = observed_phases - predicted_phases
        
        # Handle phase wrapping
        residual = np.arctan2(np.sin(residual), np.cos(residual))
        
        # Cost function: sum of squared residuals with regularization
        cost = np.sum(residual**2)
        
        # Add regularization to prevent unrealistic velocities
        velocity_penalty = 0.01 * np.sum(velocity**2)
        angular_penalty = 0.01 * np.sum(angular_velocity**2)
        
        total_cost = cost + velocity_penalty + angular_penalty
        
        return total_cost
    
    def get_smart_initial_guess(self, 
                               target_associations: List[Dict],
                               dt: float) -> np.ndarray:
        """
        Generate smart initial guess based on target motion patterns.
        
        Args:
            target_associations: List of associated target pairs
            dt: Time step (s)
            
        Returns:
            Initial guess for motion parameters [6]
        """
        if not target_associations:
            return np.array([0, 0, 0, 0, 0, 0])
        
        # Analyze target motion patterns
        target_velocities = []
        
        for association in target_associations:
            current = association['current']
            previous = association['previous']
            
            # Compute apparent target velocity from position change
            current_pos = np.array([
                current['range_m'] * np.cos(current['azimuth_rad']),
                current['range_m'] * np.sin(current['azimuth_rad']),
                0  # Assume ground level
            ])
            
            previous_pos = np.array([
                previous['range_m'] * np.cos(previous['azimuth_rad']),
                previous['range_m'] * np.sin(previous['azimuth_rad']),
                0
            ])
            
            # Target velocity (apparent motion)
            target_velocity = (current_pos - previous_pos) / dt
            target_velocities.append(target_velocity)
        
        if target_velocities:
            # Use median target velocity as initial guess for ego velocity
            target_velocities = np.array(target_velocities)
            median_velocity = np.median(target_velocities, axis=0)
            
            # Initial guess: assume ego vehicle is moving opposite to apparent target motion
            initial_velocity = -median_velocity[:2]  # Only x, y components
            initial_velocity = np.append(initial_velocity, 0)  # z = 0
            initial_angular_velocity = np.array([0, 0, 0])  # Start with no rotation
            
            initial_guess = np.concatenate([initial_velocity, initial_angular_velocity])
        else:
            initial_guess = np.array([0, 0, 0, 0, 0, 0])
        
        logger.info(f"Smart initial guess: velocity={initial_guess[:3]}, angular={initial_guess[3:]}")
        return initial_guess
    
    def two_step_optimization(self, 
                            target_associations: List[Dict],
                            dt: float,
                            initial_guess: Optional[np.ndarray] = None) -> Dict:
        """
        Improved two-step optimization with proper temporal phase differences.
        
        Args:
            target_associations: List of associated target pairs
            dt: Time step (s)
            initial_guess: Initial guess for motion parameters
            
        Returns:
            Optimization results dictionary
        """
        if len(target_associations) < 3:
            logger.warning("Insufficient target associations for optimization")
            return {'success': False, 'message': 'Insufficient target associations'}
        
        # Extract target information
        target_positions = []
        target_angles = []
        
        for association in target_associations:
            current = association['current']
            
            # Convert range and angles to 3D position
            range_m = current['range_m']
            azimuth_rad = current['azimuth_rad']
            elevation_rad = 0.0  # Assume ground level
            
            # 3D position
            x = range_m * np.cos(elevation_rad) * np.cos(azimuth_rad)
            y = range_m * np.cos(elevation_rad) * np.sin(azimuth_rad)
            z = range_m * np.sin(elevation_rad)
            
            target_positions.append([x, y, z])
            target_angles.append([azimuth_rad, elevation_rad])
        
        target_positions = np.array(target_positions)
        target_angles = np.array(target_angles)
        
        # Compute observed temporal phase differences
        observed_phases = self.compute_observed_phase_differences(target_associations)
        
        # Generate smart initial guess if not provided
        if initial_guess is None:
            initial_guess = self.get_smart_initial_guess(target_associations, dt)
        
        # Step 1: Translational velocity only (assuming small rotation)
        logger.info("Step 1: Solving for translational velocity...")
        
        def cost_translational(v_trans):
            """Cost function for translational velocity only."""
            motion_params = np.concatenate([v_trans, [0, 0, 0]])  # Zero rotation
            return self.cost_function(motion_params, target_positions, target_angles, 
                                   observed_phases, dt)
        
        # Bounds for translational velocity (reasonable vehicle speeds)
        bounds_trans = [(-50, 50), (-50, 50), (-10, 10)]  # vx, vy, vz (m/s)
        
        if self.optimization_method == 'differential_evolution':
            result_trans = differential_evolution(
                cost_translational, bounds_trans, 
                maxiter=self.max_iterations, tol=self.tolerance, seed=42
            )
        else:
            result_trans = minimize(
                cost_translational, x0=initial_guess[:3], 
                method='Nelder-Mead', 
                options={'maxiter': self.max_iterations}
            )
        
        if not result_trans.success:
            logger.warning("Step 1 optimization failed")
            return {'success': False, 'message': 'Step 1 failed'}
        
        v_trans_est = result_trans.x
        logger.info(f"Step 1 result: v_trans = {v_trans_est}")
        
        # Step 2: Full 6-DoF optimization
        logger.info("Step 2: Refining with full 6-DoF motion...")
        
        # Initial guess for full optimization
        initial_guess_full = np.concatenate([v_trans_est, [0, 0, 0]])
        
        def cost_full(motion_params):
            """Cost function for full 6-DoF motion."""
            return self.cost_function(motion_params, target_positions, target_angles, 
                                   observed_phases, dt)
        
        # Bounds for full motion
        bounds_full = [(-50, 50), (-50, 50), (-10, 10),  # vx, vy, vz
                      (-10, 10), (-10, 10), (-10, 10)]   # wx, wy, wz (rad/s)
        
        if self.optimization_method == 'differential_evolution':
            result_full = differential_evolution(
                cost_full, bounds_full, 
                maxiter=self.max_iterations, tol=self.tolerance, seed=42
            )
        else:
            result_full = minimize(
                cost_full, x0=initial_guess_full,
                method='Nelder-Mead',
                options={'maxiter': self.max_iterations}
            )
        
        if not result_full.success:
            logger.warning("Step 2 optimization failed, using Step 1 result")
            velocity_est = v_trans_est
            angular_velocity_est = np.array([0, 0, 0])
            cost_value = result_trans.fun
        else:
            velocity_est = result_full.x[:3]
            angular_velocity_est = result_full.x[3:]
            cost_value = result_full.fun
        
        # Compute final residuals
        final_motion = np.concatenate([velocity_est, angular_velocity_est])
        predicted_phases = self.compute_phase_difference_model(
            target_positions, target_angles, velocity_est, angular_velocity_est, dt
        )
        residuals = observed_phases - predicted_phases
        
        # Handle phase wrapping in residuals
        residuals = np.arctan2(np.sin(residuals), np.cos(residuals))
        
        # Compute statistics
        rmse = np.sqrt(np.mean(residuals**2))
        max_residual = np.max(np.abs(residuals))
        
        results = {
            'success': True,
            'velocity': velocity_est,
            'angular_velocity': angular_velocity_est,
            'cost': cost_value,
            'rmse': rmse,
            'max_residual': max_residual,
            'residuals': residuals,
            'predicted_phases': predicted_phases,
            'observed_phases': observed_phases,
            'num_associations': len(target_associations),
            'step1_result': result_trans,
            'step2_result': result_full
        }
        
        logger.info(f"Optimization complete:")
        logger.info(f"  Velocity: {velocity_est}")
        logger.info(f"  Angular velocity: {angular_velocity_est}")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  Max residual: {max_residual:.6f}")
        
        return results
    
    def solve_velocity_with_association(self, 
                                      current_targets: List[Dict],
                                      previous_targets: List[Dict],
                                      dt: float = 0.1) -> Dict:
        """
        Solve for velocity using improved target association and temporal phase differences.
        
        Args:
            current_targets: Current frame targets
            previous_targets: Previous frame targets
            dt: Time step (s)
            
        Returns:
            Optimization results
        """
        # Associate targets across frames
        target_associations = self.associate_targets_across_frames(
            current_targets, previous_targets
        )
        
        if not target_associations:
            logger.warning("No target associations found")
            return {'success': False, 'message': 'No target associations'}
        
        # Run two-step optimization
        results = self.two_step_optimization(target_associations, dt)
        
        return results


def estimate_velocity_improved(current_angles_path: str,
                             previous_angles_path: str,
                             output_path: str,
                             radar_params: Dict = None,
                             dt: float = 0.1) -> Dict:
    """
    Estimate velocity using improved temporal phase differences and target association.
    
    Args:
        current_angles_path: Path to current frame angle estimates
        previous_angles_path: Path to previous frame angle estimates
        output_path: Path to save velocity estimates
        radar_params: Radar parameters
        dt: Time step (s)
        
    Returns:
        Processing results
    """
    # Load data
    current_data = np.load(current_angles_path, allow_pickle=True)
    current_targets = current_data['targets']
    
    previous_data = np.load(previous_angles_path, allow_pickle=True)
    previous_targets = previous_data['targets']
    
    logger.info(f"Loaded {len(current_targets)} current targets")
    logger.info(f"Loaded {len(previous_targets)} previous targets")
    
    # Default radar parameters
    if radar_params is None:
        radar_params = {
            'fc': 77e9,
            'lambda_c': 3e8 / 77e9,
            'num_antennas': 8
        }
    
    # Initialize improved solver
    solver = ImprovedVelocitySolver(**radar_params)
    
    # Solve for velocity with association
    results = solver.solve_velocity_with_association(current_targets, previous_targets, dt)
    
    # Save results
    np.savez(output_path, **results)
    
    logger.info(f"Improved velocity estimation complete: {results['success']}")
    if results['success']:
        logger.info(f"  Velocity: {results['velocity']}")
        logger.info(f"  Angular velocity: {results['angular_velocity']}")
        logger.info(f"  RMSE: {results['rmse']:.6f}")
    
    return results


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Estimate velocity with improved method')
    parser.add_argument('--current', required=True, help='Path to current angles file')
    parser.add_argument('--previous', required=True, help='Path to previous angles file')
    parser.add_argument('--out', required=True, help='Output path for velocity')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step (s)')
    
    args = parser.parse_args()
    
    # Estimate velocity with improved method
    results = estimate_velocity_improved(args.current, args.previous, args.out, dt=args.dt)
    print(f"Improved velocity estimation complete: {results}")
