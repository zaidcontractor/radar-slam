"""
Pose Integration Module

This module implements velocity-to-pose integration for trajectory generation
following the approach described in "3-D Ego-Motion Estimation Using Multi-Channel FMCW Radar".

The integration handles:
1. Velocity integration to position
2. Angular velocity integration to orientation
3. Coordinate frame transformations
4. Trajectory smoothing and regularization
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.spatial.transform import Rotation
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PoseIntegrator:
    """
    Integrates velocity estimates to generate pose trajectories.
    
    Handles both translational and rotational motion integration
    with proper coordinate frame management.
    """
    
    def __init__(self,
                 initial_position: np.ndarray = np.array([0, 0, 0]),
                 initial_orientation: np.ndarray = np.array([0, 0, 0]),
                 coordinate_frame: str = 'body',
                 integration_method: str = 'trapezoidal',
                 smoothing: bool = True,
                 smoothing_window: int = 5):
        """
        Initialize pose integrator.
        
        Args:
            initial_position: Initial position [x, y, z] (m)
            initial_orientation: Initial orientation [roll, pitch, yaw] (rad)
            coordinate_frame: Coordinate frame ('body', 'world')
            integration_method: Integration method ('trapezoidal', 'euler')
            smoothing: Whether to apply smoothing
            smoothing_window: Smoothing window size
        """
        self.initial_position = np.array(initial_position)
        self.initial_orientation = np.array(initial_orientation)
        self.coordinate_frame = coordinate_frame
        self.integration_method = integration_method
        self.smoothing = smoothing
        self.smoothing_window = smoothing_window
        
        # Initialize state
        self.current_position = self.initial_position.copy()
        self.current_orientation = self.initial_orientation.copy()
        self.current_rotation = Rotation.from_euler('xyz', self.initial_orientation)
        
        logger.info(f"Initialized pose integrator:")
        logger.info(f"  Initial position: {self.initial_position}")
        logger.info(f"  Initial orientation: {np.degrees(self.initial_orientation)}°")
        logger.info(f"  Coordinate frame: {coordinate_frame}")
        logger.info(f"  Integration method: {integration_method}")
    
    def integrate_translational_velocity(self, 
                                       velocities: np.ndarray,
                                       timestamps: np.ndarray) -> np.ndarray:
        """
        Integrate translational velocity to position.
        
        Args:
            velocities: Velocity vectors [N, 3] (vx, vy, vz)
            timestamps: Time stamps [N] (s)
            
        Returns:
            Position trajectory [N, 3] (x, y, z)
        """
        N = len(velocities)
        
        if self.integration_method == 'trapezoidal':
            # Trapezoidal integration
            dt = np.diff(timestamps)
            positions = np.zeros((N, 3))
            positions[0] = self.initial_position
            
            for i in range(1, N):
                # Trapezoidal rule
                positions[i] = positions[i-1] + 0.5 * dt[i-1] * (velocities[i-1] + velocities[i])
        
        elif self.integration_method == 'euler':
            # Euler integration
            dt = np.diff(timestamps)
            positions = np.zeros((N, 3))
            positions[0] = self.initial_position
            
            for i in range(1, N):
                positions[i] = positions[i-1] + dt[i-1] * velocities[i-1]
        
        else:
            raise ValueError(f"Unknown integration method: {self.integration_method}")
        
        # Apply smoothing if enabled
        if self.smoothing and N > self.smoothing_window:
            from scipy.ndimage import uniform_filter1d
            for i in range(3):
                positions[:, i] = uniform_filter1d(positions[:, i], 
                                                size=self.smoothing_window, mode='nearest')
        
        return positions
    
    def integrate_angular_velocity(self, 
                                 angular_velocities: np.ndarray,
                                 timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate angular velocity to orientation.
        
        Args:
            angular_velocities: Angular velocity vectors [N, 3] (wx, wy, wz)
            timestamps: Time stamps [N] (s)
            
        Returns:
            Tuple of (orientation_trajectory, rotation_matrices)
        """
        N = len(angular_velocities)
        
        # Initialize orientation arrays
        orientations = np.zeros((N, 3))  # [roll, pitch, yaw]
        orientations[0] = self.initial_orientation
        
        # Initialize rotation matrices
        rotations = np.zeros((N, 3, 3))
        rotations[0] = self.current_rotation.as_matrix()
        
        # Integration
        dt = np.diff(timestamps)
        
        for i in range(1, N):
            # Angular velocity vector
            omega = angular_velocities[i-1]
            
            # Magnitude and axis
            omega_mag = np.linalg.norm(omega)
            
            if omega_mag > 1e-12:  # Avoid division by zero
                # Axis of rotation
                axis = omega / omega_mag
                
                # Rotation angle
                angle = omega_mag * dt[i-1]
                
                # Create rotation from axis-angle
                rotation_increment = Rotation.from_rotvec(axis * angle)
                
                # Update rotation
                new_rotation = Rotation.from_matrix(rotations[i-1]) * rotation_increment
                rotations[i] = new_rotation.as_matrix()
                
                # Convert to Euler angles
                orientations[i] = new_rotation.as_euler('xyz')
            else:
                # No rotation
                rotations[i] = rotations[i-1]
                orientations[i] = orientations[i-1]
        
        return orientations, rotations
    
    def integrate_pose(self, 
                      velocities: np.ndarray,
                      angular_velocities: np.ndarray,
                      timestamps: np.ndarray) -> Dict:
        """
        Integrate both translational and rotational velocities to full pose.
        
        Args:
            velocities: Velocity vectors [N, 3] (vx, vy, vz)
            angular_velocities: Angular velocity vectors [N, 3] (wx, wy, wz)
            timestamps: Time stamps [N] (s)
            
        Returns:
            Pose trajectory dictionary
        """
        N = len(velocities)
        
        if len(angular_velocities) != N or len(timestamps) != N:
            raise ValueError("All input arrays must have the same length")
        
        logger.info(f"Integrating pose for {N} time steps")
        
        # Integrate translational motion
        positions = self.integrate_translational_velocity(velocities, timestamps)
        
        # Integrate rotational motion
        orientations, rotations = self.integrate_angular_velocity(angular_velocities, timestamps)
        
        # Compute trajectory statistics
        total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        total_rotation = np.sum(np.linalg.norm(angular_velocities, axis=1) * np.diff(timestamps))
        
        # Create trajectory dictionary
        trajectory = {
            'timestamps': timestamps,
            'positions': positions,
            'orientations': orientations,
            'rotations': rotations,
            'velocities': velocities,
            'angular_velocities': angular_velocities,
            'total_distance': total_distance,
            'total_rotation': total_rotation,
            'duration': timestamps[-1] - timestamps[0],
            'num_points': N
        }
        
        logger.info(f"Pose integration complete:")
        logger.info(f"  Total distance: {total_distance:.2f} m")
        logger.info(f"  Total rotation: {total_rotation:.2f} rad")
        logger.info(f"  Duration: {trajectory['duration']:.2f} s")
        
        return trajectory
    
    def transform_to_world_frame(self, 
                               trajectory: Dict,
                               initial_world_pose: Optional[Dict] = None) -> Dict:
        """
        Transform trajectory to world coordinate frame.
        
        Args:
            trajectory: Body frame trajectory
            initial_world_pose: Initial world frame pose
            
        Returns:
            World frame trajectory
        """
        if initial_world_pose is None:
            # Use identity transformation
            initial_world_pose = {
                'position': np.array([0, 0, 0]),
                'orientation': np.array([0, 0, 0])
            }
        
        # Extract initial world pose
        world_pos_0 = initial_world_pose['position']
        world_rot_0 = Rotation.from_euler('xyz', initial_world_pose['orientation'])
        
        # Transform trajectory
        world_positions = np.zeros_like(trajectory['positions'])
        world_orientations = np.zeros_like(trajectory['orientations'])
        world_rotations = np.zeros_like(trajectory['rotations'])
        
        for i in range(len(trajectory['positions'])):
            # Transform position
            body_pos = trajectory['positions'][i]
            world_positions[i] = world_pos_0 + world_rot_0.apply(body_pos)
            
            # Transform orientation
            body_rot = Rotation.from_matrix(trajectory['rotations'][i])
            world_rot = world_rot_0 * body_rot
            world_orientations[i] = world_rot.as_euler('xyz')
            world_rotations[i] = world_rot.as_matrix()
        
        # Create world frame trajectory
        world_trajectory = trajectory.copy()
        world_trajectory['positions'] = world_positions
        world_trajectory['orientations'] = world_orientations
        world_trajectory['rotations'] = world_rotations
        world_trajectory['coordinate_frame'] = 'world'
        
        return world_trajectory
    
    def visualize_trajectory(self, 
                           trajectory: Dict,
                           save_path: Optional[str] = None,
                           show_orientation: bool = True) -> None:
        """
        Visualize pose trajectory.
        
        Args:
            trajectory: Trajectory dictionary
            save_path: Path to save plot
            show_orientation: Whether to show orientation arrows
        """
        positions = trajectory['positions']
        orientations = trajectory['orientations']
        timestamps = trajectory['timestamps']
        
        fig = plt.figure(figsize=(15, 10))
        
        # 3D trajectory
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
        ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                   color='green', s=100, label='Start')
        ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                   color='red', s=100, label='End')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory')
        ax1.legend()
        
        # 2D trajectory (top view)
        ax2 = fig.add_subplot(222)
        ax2.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
        ax2.scatter(positions[0, 0], positions[0, 1], color='green', s=100, label='Start')
        ax2.scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, label='End')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Top View')
        ax2.legend()
        ax2.grid(True)
        ax2.axis('equal')
        
        # Position vs time
        ax3 = fig.add_subplot(223)
        ax3.plot(timestamps, positions[:, 0], label='X', linewidth=2)
        ax3.plot(timestamps, positions[:, 1], label='Y', linewidth=2)
        ax3.plot(timestamps, positions[:, 2], label='Z', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Position (m)')
        ax3.set_title('Position vs Time')
        ax3.legend()
        ax3.grid(True)
        
        # Orientation vs time
        ax4 = fig.add_subplot(224)
        ax4.plot(timestamps, np.degrees(orientations[:, 0]), label='Roll', linewidth=2)
        ax4.plot(timestamps, np.degrees(orientations[:, 1]), label='Pitch', linewidth=2)
        ax4.plot(timestamps, np.degrees(orientations[:, 2]), label='Yaw', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Orientation (deg)')
        ax4.set_title('Orientation vs Time')
        ax4.legend()
        ax4.grid(True)
        
        # Add trajectory statistics
        stats_text = f"Distance: {trajectory['total_distance']:.2f} m\n"
        stats_text += f"Rotation: {trajectory['total_rotation']:.2f} rad\n"
        stats_text += f"Duration: {trajectory['duration']:.2f} s\n"
        stats_text += f"Points: {trajectory['num_points']}"
        
        fig.text(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_trajectory(self, 
                       trajectory: Dict,
                       output_path: str) -> None:
        """
        Save trajectory to file.
        
        Args:
            trajectory: Trajectory dictionary
            output_path: Output file path
        """
        # Save as numpy file
        np.savez(output_path, **trajectory)
        
        # Also save as text file for compatibility
        txt_path = output_path.replace('.npz', '.txt')
        with open(txt_path, 'w') as f:
            f.write("# Trajectory data\n")
            f.write("# Format: timestamp, x, y, z, roll, pitch, yaw\n")
            
            for i in range(len(trajectory['timestamps'])):
                t = trajectory['timestamps'][i]
                pos = trajectory['positions'][i]
                ori = trajectory['orientations'][i]
                
                f.write(f"{t:.6f}, {pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}, "
                       f"{ori[0]:.6f}, {ori[1]:.6f}, {ori[2]:.6f}\n")
        
        logger.info(f"Trajectory saved to {output_path}")


def integrate_velocities_to_pose(velocities_path: str,
                               output_path: str,
                               initial_pose: Optional[Dict] = None,
                               dt: float = 0.1) -> Dict:
    """
    Integrate velocity estimates to generate pose trajectory.
    
    Args:
        velocities_path: Path to velocity estimates file
        output_path: Path to save trajectory
        initial_pose: Initial pose dictionary
        dt: Time step (s)
        
    Returns:
        Trajectory dictionary
    """
    # Load velocity data
    velocity_data = np.load(velocities_path, allow_pickle=True)
    
    # Extract velocity arrays
    velocities = velocity_data['velocity']
    angular_velocities = velocity_data['angular_velocity']
    
    # Create timestamps
    timestamps = np.arange(len(velocities)) * dt
    
    logger.info(f"Loaded velocity data: {len(velocities)} points")
    
    # Initialize integrator
    integrator = PoseIntegrator()
    
    # Integrate to pose
    trajectory = integrator.integrate_pose(velocities, angular_velocities, timestamps)
    
    # Transform to world frame if initial pose provided
    if initial_pose is not None:
        trajectory = integrator.transform_to_world_frame(trajectory, initial_pose)
    
    # Save trajectory
    integrator.save_trajectory(trajectory, output_path)
    
    logger.info(f"Pose integration complete: {output_path}")
    
    return trajectory


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrate velocities to pose')
    parser.add_argument('--velocities', required=True, help='Path to velocities file')
    parser.add_argument('--out', required=True, help='Output path for trajectory')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step (s)')
    
    args = parser.parse_args()
    
    # Integrate velocities
    trajectory = integrate_velocities_to_pose(args.velocities, args.out, dt=args.dt)
    print(f"Pose integration complete: {len(trajectory['positions'])} points")
