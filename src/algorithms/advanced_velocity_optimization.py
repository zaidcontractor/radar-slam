"""
Advanced Velocity Optimization Module

This module implements enhanced velocity optimization with:
1. Regularization terms to cost function
2. Adaptive bounds based on vehicle dynamics
3. Multiple optimization runs with different initializations
4. Real-time processing optimizations
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution, basinhopping
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Optional, Callable
import logging
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

logger = logging.getLogger(__name__)

class AdvancedVelocityOptimizer:
    """
    Advanced velocity optimization with regularization and adaptive bounds.
    
    Features:
    - Regularization terms for stable optimization
    - Adaptive bounds based on vehicle dynamics
    - Multiple optimization runs for robustness
    - Real-time processing optimizations
    """
    
    def __init__(self,
                 fc: float = 77e9,
                 lambda_c: float = None,
                 num_antennas: int = 8,
                 antenna_spacing: float = None,
                 max_velocity: float = 50.0,  # m/s
                 max_angular_velocity: float = 10.0,  # rad/s
                 regularization_weight: float = 0.01,
                 num_optimization_runs: int = 3,
                 use_parallel: bool = True):
        """
        Initialize advanced velocity optimizer.
        
        Args:
            fc: Carrier frequency (Hz)
            lambda_c: Wavelength (m)
            num_antennas: Number of antenna elements
            antenna_spacing: Antenna spacing (m)
            max_velocity: Maximum expected velocity (m/s)
            max_angular_velocity: Maximum expected angular velocity (rad/s)
            regularization_weight: Weight for regularization terms
            num_optimization_runs: Number of optimization runs for robustness
            use_parallel: Whether to use parallel processing
        """
        self.fc = fc
        self.c = 3e8  # Speed of light
        self.lambda_c = lambda_c or (self.c / self.fc)
        self.num_antennas = num_antennas
        self.antenna_spacing = antenna_spacing or (self.lambda_c / 2)
        self.max_velocity = max_velocity
        self.max_angular_velocity = max_angular_velocity
        self.regularization_weight = regularization_weight
        self.num_optimization_runs = num_optimization_runs
        self.use_parallel = use_parallel
        
        # Antenna positions
        self.antenna_positions = np.arange(self.num_antennas) * self.antenna_spacing
        
        # Vehicle dynamics constraints
        self.velocity_history = []
        self.angular_velocity_history = []
        self.adaptive_bounds = self._initialize_adaptive_bounds()
        
        logger.info(f"Initialized advanced velocity optimizer:")
        logger.info(f"  Max velocity: {max_velocity} m/s")
        logger.info(f"  Max angular velocity: {max_angular_velocity} rad/s")
        logger.info(f"  Regularization weight: {regularization_weight}")
        logger.info(f"  Optimization runs: {num_optimization_runs}")
        logger.info(f"  Parallel processing: {use_parallel}")
    
    def _initialize_adaptive_bounds(self) -> Dict:
        """Initialize adaptive bounds based on vehicle dynamics."""
        return {
            'velocity_bounds': [(-self.max_velocity, self.max_velocity)] * 3,
            'angular_velocity_bounds': [(-self.max_angular_velocity, self.max_angular_velocity)] * 3,
            'acceleration_bounds': [(-20, 20)] * 3,  # m/s²
            'angular_acceleration_bounds': [(-5, 5)] * 3  # rad/s²
        }
    
    def update_adaptive_bounds(self, 
                              current_velocity: np.ndarray,
                              current_angular_velocity: np.ndarray,
                              dt: float = 0.1) -> None:
        """
        Update adaptive bounds based on current motion state.
        
        Args:
            current_velocity: Current velocity estimate
            current_angular_velocity: Current angular velocity estimate
            dt: Time step
        """
        # Store history
        self.velocity_history.append(current_velocity.copy())
        self.angular_velocity_history.append(current_angular_velocity.copy())
        
        # Keep only recent history
        max_history = 10
        if len(self.velocity_history) > max_history:
            self.velocity_history = self.velocity_history[-max_history:]
            self.angular_velocity_history = self.angular_velocity_history[-max_history:]
        
        if len(self.velocity_history) >= 2:
            # Compute velocity and angular velocity changes
            vel_changes = np.diff(self.velocity_history, axis=0)
            ang_vel_changes = np.diff(self.angular_velocity_history, axis=0)
            
            # Estimate acceleration bounds
            max_acceleration = np.max(np.abs(vel_changes) / dt) if dt > 0 else 20.0
            max_angular_acceleration = np.max(np.abs(ang_vel_changes) / dt) if dt > 0 else 5.0
            
            # Update bounds with safety margins
            safety_factor = 2.0
            self.adaptive_bounds['acceleration_bounds'] = [
                (-max_acceleration * safety_factor, max_acceleration * safety_factor)
            ] * 3
            self.adaptive_bounds['angular_acceleration_bounds'] = [
                (-max_angular_acceleration * safety_factor, max_angular_acceleration * safety_factor)
            ] * 3
            
            # Update velocity bounds based on current motion
            current_speed = np.linalg.norm(current_velocity)
            if current_speed > 0:
                # Expand bounds in direction of motion
                direction = current_velocity / current_speed
                velocity_expansion = min(10.0, current_speed * 0.5)  # Expand by up to 10 m/s
                
                for i in range(3):
                    if direction[i] > 0:
                        self.adaptive_bounds['velocity_bounds'][i] = (
                            -self.max_velocity,
                            min(self.max_velocity, current_velocity[i] + velocity_expansion)
                        )
                    else:
                        self.adaptive_bounds['velocity_bounds'][i] = (
                            max(-self.max_velocity, current_velocity[i] - velocity_expansion),
                            self.max_velocity
                        )
    
    def compute_regularized_cost_function(self, 
                                        motion_params: np.ndarray,
                                        target_positions: np.ndarray,
                                        target_angles: np.ndarray,
                                        observed_phases: np.ndarray,
                                        dt: float,
                                        previous_motion: Optional[np.ndarray] = None) -> float:
        """
        Compute regularized cost function for velocity optimization.
        
        Args:
            motion_params: Motion parameters [6] (vx, vy, vz, wx, wy, wz)
            target_positions: Target positions [N, 3]
            target_angles: Target angles [N, 2]
            observed_phases: Observed temporal phase differences [N]
            dt: Time step (s)
            previous_motion: Previous motion parameters for temporal regularization
            
        Returns:
            Regularized cost function value
        """
        # Split motion parameters
        velocity = motion_params[:3]
        angular_velocity = motion_params[3:]
        
        # Compute predicted temporal phase differences
        predicted_phases = self._compute_phase_difference_model(
            target_positions, target_angles, velocity, angular_velocity, dt
        )
        
        # Compute residual
        residual = observed_phases - predicted_phases
        
        # Handle phase wrapping
        residual = np.arctan2(np.sin(residual), np.cos(residual))
        
        # Base cost function
        base_cost = np.sum(residual**2)
        
        # Regularization terms
        regularization_cost = 0.0
        
        # 1. Velocity magnitude regularization (prefer reasonable speeds)
        velocity_magnitude = np.linalg.norm(velocity)
        if velocity_magnitude > self.max_velocity * 0.8:  # Penalize high speeds
            regularization_cost += self.regularization_weight * (velocity_magnitude - self.max_velocity * 0.8)**2
        
        # 2. Angular velocity magnitude regularization
        angular_velocity_magnitude = np.linalg.norm(angular_velocity)
        if angular_velocity_magnitude > self.max_angular_velocity * 0.8:
            regularization_cost += self.regularization_weight * (angular_velocity_magnitude - self.max_angular_velocity * 0.8)**2
        
        # 3. Temporal smoothness regularization (if previous motion available)
        if previous_motion is not None:
            motion_change = motion_params - previous_motion
            temporal_penalty = np.sum(motion_change**2)
            regularization_cost += self.regularization_weight * 0.1 * temporal_penalty
        
        # 4. Physical constraints regularization
        # Penalize unrealistic combinations (e.g., high velocity with high angular velocity)
        if velocity_magnitude > 20 and angular_velocity_magnitude > 5:
            unrealistic_penalty = (velocity_magnitude - 20) * (angular_velocity_magnitude - 5)
            regularization_cost += self.regularization_weight * 0.01 * unrealistic_penalty
        
        # 5. Z-velocity regularization (prefer small vertical motion)
        z_velocity_penalty = velocity[2]**2
        regularization_cost += self.regularization_weight * 10.0 * z_velocity_penalty
        
        total_cost = base_cost + regularization_cost
        
        return total_cost
    
    def _compute_phase_difference_model(self, 
                                      target_positions: np.ndarray,
                                      target_angles: np.ndarray,
                                      velocity: np.ndarray,
                                      angular_velocity: np.ndarray,
                                      dt: float) -> np.ndarray:
        """Compute predicted temporal phase differences from motion model."""
        N = len(target_positions)
        predicted_phases = np.zeros(N)
        
        for i in range(N):
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
            cross_product = np.cross(angular_velocity, pos)
            relative_velocity = velocity + cross_product
            
            # Radial velocity component
            radial_velocity = np.dot(relative_velocity, direction)
            
            # Temporal phase difference
            phase_diff = (4 * np.pi * radial_velocity * dt) / self.lambda_c
            predicted_phases[i] = phase_diff
        
        return predicted_phases
    
    def generate_multiple_initial_guesses(self, 
                                        target_associations: List[Dict],
                                        dt: float) -> List[np.ndarray]:
        """
        Generate multiple initial guesses for robust optimization.
        
        Args:
            target_associations: List of associated target pairs
            dt: Time step (s)
            
        Returns:
            List of initial guesses
        """
        initial_guesses = []
        
        # 1. Smart initial guess based on target motion
        smart_guess = self._generate_smart_initial_guess(target_associations, dt)
        initial_guesses.append(smart_guess)
        
        # 2. Zero motion guess
        zero_guess = np.zeros(6)
        initial_guesses.append(zero_guess)
        
        # 3. Random guesses within reasonable bounds
        for _ in range(self.num_optimization_runs - 2):
            random_guess = np.array([
                np.random.uniform(-self.max_velocity * 0.5, self.max_velocity * 0.5),  # vx
                np.random.uniform(-self.max_velocity * 0.5, self.max_velocity * 0.5),  # vy
                np.random.uniform(-5, 5),  # vz
                np.random.uniform(-self.max_angular_velocity * 0.5, self.max_angular_velocity * 0.5),  # wx
                np.random.uniform(-self.max_angular_velocity * 0.5, self.max_angular_velocity * 0.5),  # wy
                np.random.uniform(-self.max_angular_velocity * 0.5, self.max_angular_velocity * 0.5)   # wz
            ])
            initial_guesses.append(random_guess)
        
        return initial_guesses
    
    def _generate_smart_initial_guess(self, 
                                     target_associations: List[Dict],
                                     dt: float) -> np.ndarray:
        """Generate smart initial guess based on target motion patterns."""
        if not target_associations:
            return np.zeros(6)
        
        # Analyze target motion patterns
        target_velocities = []
        
        for association in target_associations:
            current = association['current']
            previous = association['previous']
            
            # Compute apparent target velocity
            current_pos = np.array([
                current['range_m'] * np.cos(current['azimuth_rad']),
                current['range_m'] * np.sin(current['azimuth_rad']),
                0
            ])
            
            previous_pos = np.array([
                previous['range_m'] * np.cos(previous['azimuth_rad']),
                previous['range_m'] * np.sin(previous['azimuth_rad']),
                0
            ])
            
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
            initial_guess = np.zeros(6)
        
        return initial_guess
    
    def run_single_optimization(self, 
                               initial_guess: np.ndarray,
                               target_positions: np.ndarray,
                               target_angles: np.ndarray,
                               observed_phases: np.ndarray,
                               dt: float,
                               previous_motion: Optional[np.ndarray] = None) -> Dict:
        """
        Run a single optimization with given initial guess.
        
        Args:
            initial_guess: Initial guess for motion parameters
            target_positions: Target positions [N, 3]
            target_angles: Target angles [N, 2]
            observed_phases: Observed temporal phase differences [N]
            dt: Time step (s)
            previous_motion: Previous motion parameters
            
        Returns:
            Optimization results
        """
        # Define cost function
        def cost_function(motion_params):
            return self.compute_regularized_cost_function(
                motion_params, target_positions, target_angles, 
                observed_phases, dt, previous_motion
            )
        
        # Get adaptive bounds
        bounds = (self.adaptive_bounds['velocity_bounds'] + 
                self.adaptive_bounds['angular_velocity_bounds'])
        
        # Run optimization
        try:
            result = differential_evolution(
                cost_function, bounds, 
                maxiter=1000, tol=1e-6, seed=42,
                workers=1  # Single thread for this run
            )
            
            if result.success:
                return {
                    'success': True,
                    'motion_params': result.x,
                    'cost': result.fun,
                    'iterations': result.nit,
                    'initial_guess': initial_guess
                }
            else:
                return {
                    'success': False,
                    'motion_params': initial_guess,
                    'cost': float('inf'),
                    'iterations': 0,
                    'initial_guess': initial_guess,
                    'error': 'Optimization failed'
                }
        except Exception as e:
            return {
                'success': False,
                'motion_params': initial_guess,
                'cost': float('inf'),
                'iterations': 0,
                'initial_guess': initial_guess,
                'error': str(e)
            }
    
    def run_robust_optimization(self, 
                               target_associations: List[Dict],
                               dt: float,
                               previous_motion: Optional[np.ndarray] = None) -> Dict:
        """
        Run robust optimization with multiple initial guesses.
        
        Args:
            target_associations: List of associated target pairs
            dt: Time step (s)
            previous_motion: Previous motion parameters
            
        Returns:
            Best optimization results
        """
        if len(target_associations) < 3:
            logger.warning("Insufficient target associations for optimization")
            return {'success': False, 'message': 'Insufficient target associations'}
        
        # Extract target information
        target_positions = []
        target_angles = []
        
        for association in target_associations:
            current = association['current']
            
            range_m = current['range_m']
            azimuth_rad = current['azimuth_rad']
            elevation_rad = 0.0  # Assume ground level
            
            x = range_m * np.cos(elevation_rad) * np.cos(azimuth_rad)
            y = range_m * np.cos(elevation_rad) * np.sin(azimuth_rad)
            z = range_m * np.sin(elevation_rad)
            
            target_positions.append([x, y, z])
            target_angles.append([azimuth_rad, elevation_rad])
        
        target_positions = np.array(target_positions)
        target_angles = np.array(target_angles)
        
        # Compute observed temporal phase differences
        observed_phases = np.array([a['temporal_phase_diff'] for a in target_associations])
        
        # Generate multiple initial guesses
        initial_guesses = self.generate_multiple_initial_guesses(target_associations, dt)
        
        # Run multiple optimizations
        if self.use_parallel and len(initial_guesses) > 1:
            # Parallel optimization
            with ThreadPoolExecutor(max_workers=min(len(initial_guesses), mp.cpu_count())) as executor:
                futures = []
                for initial_guess in initial_guesses:
                    future = executor.submit(
                        self.run_single_optimization,
                        initial_guess, target_positions, target_angles,
                        observed_phases, dt, previous_motion
                    )
                    futures.append(future)
                
                results = [future.result() for future in futures]
        else:
            # Sequential optimization
            results = []
            for initial_guess in initial_guesses:
                result = self.run_single_optimization(
                    initial_guess, target_positions, target_angles,
                    observed_phases, dt, previous_motion
                )
                results.append(result)
        
        # Select best result
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            logger.warning("All optimization runs failed")
            return {'success': False, 'message': 'All optimization runs failed'}
        
        # Choose result with lowest cost
        best_result = min(successful_results, key=lambda x: x['cost'])
        
        # Extract motion parameters
        motion_params = best_result['motion_params']
        velocity = motion_params[:3]
        angular_velocity = motion_params[3:]
        
        # Compute final residuals
        predicted_phases = self._compute_phase_difference_model(
            target_positions, target_angles, velocity, angular_velocity, dt
        )
        residuals = observed_phases - predicted_phases
        residuals = np.arctan2(np.sin(residuals), np.cos(residuals))
        
        # Update adaptive bounds
        self.update_adaptive_bounds(velocity, angular_velocity, dt)
        
        # Compute statistics
        rmse = np.sqrt(np.mean(residuals**2))
        max_residual = np.max(np.abs(residuals))
        
        return {
            'success': True,
            'velocity': velocity,
            'angular_velocity': angular_velocity,
            'cost': best_result['cost'],
            'rmse': rmse,
            'max_residual': max_residual,
            'residuals': residuals,
            'predicted_phases': predicted_phases,
            'observed_phases': observed_phases,
            'num_associations': len(target_associations),
            'num_optimization_runs': len(results),
            'successful_runs': len(successful_results),
            'best_initial_guess': best_result['initial_guess'],
            'all_results': results
        }


def optimize_velocity_advanced(target_associations: List[Dict],
                              dt: float = 0.1,
                              radar_params: Dict = None,
                              previous_motion: Optional[np.ndarray] = None) -> Dict:
    """
    Optimize velocity using advanced methods with regularization and multiple runs.
    
    Args:
        target_associations: List of associated target pairs
        dt: Time step (s)
        radar_params: Radar parameters
        previous_motion: Previous motion parameters for temporal regularization
        
    Returns:
        Optimization results
    """
    # Default radar parameters
    if radar_params is None:
        radar_params = {
            'fc': 77e9,
            'lambda_c': 3e8 / 77e9,
            'num_antennas': 8
        }
    
    # Initialize advanced optimizer
    optimizer = AdvancedVelocityOptimizer(**radar_params)
    
    # Run robust optimization
    results = optimizer.run_robust_optimization(target_associations, dt, previous_motion)
    
    logger.info(f"Advanced velocity optimization complete: {results['success']}")
    if results['success']:
        logger.info(f"  Velocity: {results['velocity']}")
        logger.info(f"  Angular velocity: {results['angular_velocity']}")
        logger.info(f"  RMSE: {results['rmse']:.6f}")
        logger.info(f"  Successful runs: {results['successful_runs']}/{results['num_optimization_runs']}")
    
    return results


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced velocity optimization')
    parser.add_argument('--associations', required=True, help='Path to target associations file')
    parser.add_argument('--out', required=True, help='Output path for velocity')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step (s)')
    
    args = parser.parse_args()
    
    # Load target associations
    associations_data = np.load(args.associations, allow_pickle=True)
    target_associations = associations_data['associations']
    
    # Optimize velocity
    results = optimize_velocity_advanced(target_associations, args.dt)
    print(f"Advanced velocity optimization complete: {results}")
