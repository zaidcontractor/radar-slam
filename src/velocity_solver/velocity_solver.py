"""
Velocity Optimization Module

This module implements the two-step velocity optimization described in the paper
"3-D Ego-Motion Estimation Using Multi-Channel FMCW Radar" (Yuan et al. 2023).

The optimization solves for translational velocity [v_rx, v_ry, v_rz] and 
rotational rates [ω_x, ω_y, ω_z] using the phase-based cost function.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from typing import Dict, List, Tuple, Optional, Callable
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class VelocitySolver:
    """
    Two-step velocity optimization solver following the paper's approach.
    
    Implements the cost function f = [Y - X]^T [Y - X] where:
    - Y: Observed phase differences from RDS
    - X: Predicted phase differences from motion model
    """
    
    def __init__(self,
                 fc: float = 77e9,
                 lambda_c: float = None,
                 num_antennas: int = 8,
                 antenna_spacing: float = None,
                 optimization_method: str = 'differential_evolution',
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6):
        """
        Initialize velocity solver.
        
        Args:
            fc: Carrier frequency (Hz)
            lambda_c: Wavelength (m)
            num_antennas: Number of antenna elements
            antenna_spacing: Antenna spacing (m)
            optimization_method: Optimization method ('differential_evolution', 'nelder_mead')
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
        """
        self.fc = fc
        self.c = 3e8  # Speed of light
        self.lambda_c = lambda_c or (self.c / self.fc)
        self.num_antennas = num_antennas
        self.antenna_spacing = antenna_spacing or (self.lambda_c / 2)
        self.optimization_method = optimization_method
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Antenna positions
        self.antenna_positions = np.arange(self.num_antennas) * self.antenna_spacing
        
        logger.info(f"Initialized velocity solver:")
        logger.info(f"  Wavelength: {self.lambda_c*1000:.2f} mm")
        logger.info(f"  Optimization method: {optimization_method}")
    
    def compute_phase_difference_model(self, 
                                     target_positions: np.ndarray,
                                     target_angles: np.ndarray,
                                     velocity: np.ndarray,
                                     angular_velocity: np.ndarray,
                                     dt: float) -> np.ndarray:
        """
        Compute predicted phase differences from motion model.
        
        Args:
            target_positions: Target positions [N, 3] (x, y, z)
            target_angles: Target angles [N, 2] (azimuth, elevation)
            velocity: Translational velocity [3] (vx, vy, vz)
            angular_velocity: Angular velocity [3] (wx, wy, wz)
            dt: Time step (s)
            
        Returns:
            Predicted phase differences [N]
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
            
            # Phase difference
            phase_diff = (4 * np.pi * radial_velocity * dt) / self.lambda_c
            
            predicted_phases[i] = phase_diff
        
        return predicted_phases
    
    def compute_observed_phase_differences(self, 
                                         rds_data: np.ndarray,
                                         target_info: List[Dict]) -> np.ndarray:
        """
        Compute observed phase differences from RDS data.
        
        Args:
            rds_data: RDS matrix [num_antennas, range_bins, doppler_bins]
            target_info: List of target information dictionaries
            
        Returns:
            Observed phase differences [N]
        """
        observed_phases = []
        
        for target in target_info:
            # Extract phase from spatial signature
            spatial_signature = target['spatial_signature']
            
            # Compute phase difference between consecutive antennas
            # This is a simplified approach - in practice, you'd use temporal phase differences
            phase_diff = np.angle(spatial_signature[1] * np.conj(spatial_signature[0]))
            
            observed_phases.append(phase_diff)
        
        return np.array(observed_phases)
    
    def cost_function(self, 
                    motion_params: np.ndarray,
                    target_positions: np.ndarray,
                    target_angles: np.ndarray,
                    observed_phases: np.ndarray,
                    dt: float) -> float:
        """
        Cost function for velocity optimization.
        
        Args:
            motion_params: Motion parameters [6] (vx, vy, vz, wx, wy, wz)
            target_positions: Target positions [N, 3]
            target_angles: Target angles [N, 2]
            observed_phases: Observed phase differences [N]
            dt: Time step (s)
            
        Returns:
            Cost function value
        """
        # Split motion parameters
        velocity = motion_params[:3]
        angular_velocity = motion_params[3:]
        
        # Compute predicted phase differences
        predicted_phases = self.compute_phase_difference_model(
            target_positions, target_angles, velocity, angular_velocity, dt
        )
        
        # Compute residual
        residual = observed_phases - predicted_phases
        
        # Cost function: sum of squared residuals
        cost = np.sum(residual**2)
        
        return cost
    
    def two_step_optimization(self, 
                            target_positions: np.ndarray,
                            target_angles: np.ndarray,
                            observed_phases: np.ndarray,
                            dt: float,
                            initial_guess: Optional[np.ndarray] = None) -> Dict:
        """
        Two-step optimization as described in the paper.
        
        Step 1: Solve for translational velocity assuming small rotation
        Step 2: Refine with full 6-DoF motion
        
        Args:
            target_positions: Target positions [N, 3]
            target_angles: Target angles [N, 2]
            observed_phases: Observed phase differences [N]
            dt: Time step (s)
            initial_guess: Initial guess for motion parameters
            
        Returns:
            Optimization results dictionary
        """
        N = len(target_positions)
        
        if N < 3:
            logger.warning("Insufficient targets for optimization")
            return {'success': False, 'message': 'Insufficient targets'}
        
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
                cost_translational, x0=[0, 0, 0], 
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
        if initial_guess is None:
            initial_guess = np.concatenate([v_trans_est, [0, 0, 0]])
        
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
                cost_full, x0=initial_guess,
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
            'num_targets': N,
            'step1_result': result_trans,
            'step2_result': result_full
        }
        
        logger.info(f"Optimization complete:")
        logger.info(f"  Velocity: {velocity_est}")
        logger.info(f"  Angular velocity: {angular_velocity_est}")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  Max residual: {max_residual:.6f}")
        
        return results
    
    def solve_velocity(self, 
                      rds_data: np.ndarray,
                      target_info: List[Dict],
                      dt: float = 0.1,
                      initial_guess: Optional[np.ndarray] = None) -> Dict:
        """
        Solve for velocity from RDS data and target information.
        
        Args:
            rds_data: RDS matrix [num_antennas, range_bins, doppler_bins]
            target_info: List of target information dictionaries
            dt: Time step (s)
            initial_guess: Initial guess for motion parameters
            
        Returns:
            Optimization results
        """
        # Extract target information
        target_positions = []
        target_angles = []
        
        for target in target_info:
            # Convert range and angles to 3D position
            range_m = target['range_m']
            azimuth_rad = target['azimuth_rad']
            elevation_rad = 0.0  # Assume ground level
            
            # 3D position
            x = range_m * np.cos(elevation_rad) * np.cos(azimuth_rad)
            y = range_m * np.cos(elevation_rad) * np.sin(azimuth_rad)
            z = range_m * np.sin(elevation_rad)
            
            target_positions.append([x, y, z])
            target_angles.append([azimuth_rad, elevation_rad])
        
        target_positions = np.array(target_positions)
        target_angles = np.array(target_angles)
        
        # Compute observed phase differences
        observed_phases = self.compute_observed_phase_differences(rds_data, target_info)
        
        # Run two-step optimization
        results = self.two_step_optimization(
            target_positions, target_angles, observed_phases, dt, initial_guess
        )
        
        return results
    
    def visualize_results(self, 
                        results: Dict,
                        save_path: Optional[str] = None) -> None:
        """
        Visualize optimization results.
        
        Args:
            results: Optimization results dictionary
            save_path: Path to save plot
        """
        if not results['success']:
            logger.warning("Cannot visualize failed optimization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Velocity components
        velocity = results['velocity']
        angular_velocity = results['angular_velocity']
        
        axes[0, 0].bar(['vx', 'vy', 'vz'], velocity)
        axes[0, 0].set_ylabel('Velocity (m/s)')
        axes[0, 0].set_title('Translational Velocity')
        axes[0, 0].grid(True)
        
        axes[0, 1].bar(['wx', 'wy', 'wz'], angular_velocity)
        axes[0, 1].set_ylabel('Angular Velocity (rad/s)')
        axes[0, 1].set_title('Rotational Velocity')
        axes[0, 1].grid(True)
        
        # Plot 2: Residuals
        residuals = results['residuals']
        axes[1, 0].plot(residuals, 'o-', alpha=0.7)
        axes[1, 0].set_xlabel('Target Index')
        axes[1, 0].set_ylabel('Phase Residual (rad)')
        axes[1, 0].set_title('Phase Residuals')
        axes[1, 0].grid(True)
        
        # Plot 3: Predicted vs Observed
        predicted = results['predicted_phases']
        observed = results['observed_phases']
        axes[1, 1].scatter(observed, predicted, alpha=0.7)
        axes[1, 1].plot([observed.min(), observed.max()], 
                       [observed.min(), observed.max()], 'r--', alpha=0.5)
        axes[1, 1].set_xlabel('Observed Phase (rad)')
        axes[1, 1].set_ylabel('Predicted Phase (rad)')
        axes[1, 1].set_title('Predicted vs Observed')
        axes[1, 1].grid(True)
        
        # Add text with statistics
        stats_text = f"RMSE: {results['rmse']:.4f}\nMax Residual: {results['max_residual']:.4f}\nCost: {results['cost']:.4f}"
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def estimate_velocity_from_angles(angles_path: str,
                                rds_path: str,
                                output_path: str,
                                radar_params: Dict = None,
                                dt: float = 0.1) -> Dict:
    """
    Estimate velocity from angle estimates and RDS data.
    
    Args:
        angles_path: Path to angle estimates file
        rds_path: Path to RDS file
        output_path: Path to save velocity estimates
        radar_params: Radar parameters
        dt: Time step (s)
        
    Returns:
        Processing results
    """
    # Load data
    angles_data = np.load(angles_path, allow_pickle=True)
    target_info = angles_data['targets'].item()
    rds = np.load(rds_path)
    
    logger.info(f"Loaded {len(target_info)} targets")
    logger.info(f"RDS shape: {rds.shape}")
    
    # Default radar parameters
    if radar_params is None:
        radar_params = {
            'fc': 77e9,
            'lambda_c': 3e8 / 77e9,
            'num_antennas': 8
        }
    
    # Initialize solver
    solver = VelocitySolver(**radar_params)
    
    # Solve for velocity
    results = solver.solve_velocity(rds, target_info, dt)
    
    # Save results
    np.savez(output_path, **results)
    
    logger.info(f"Velocity estimation complete: {results['success']}")
    if results['success']:
        logger.info(f"  Velocity: {results['velocity']}")
        logger.info(f"  Angular velocity: {results['angular_velocity']}")
        logger.info(f"  RMSE: {results['rmse']:.6f}")
    
    return results


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Estimate velocity from angles')
    parser.add_argument('--angles', required=True, help='Path to angles file')
    parser.add_argument('--rds', required=True, help='Path to RDS file')
    parser.add_argument('--out', required=True, help='Output path for velocity')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step (s)')
    
    args = parser.parse_args()
    
    # Estimate velocity
    results = estimate_velocity_from_angles(args.angles, args.rds, args.out, dt=args.dt)
    print(f"Velocity estimation complete: {results}")
