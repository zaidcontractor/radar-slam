#!/usr/bin/env python3
"""
Data Visualization and Analysis Script

This script helps visualize and analyze the results from the ego-motion estimation pipeline.
It provides various plots to understand the radar data, signal processing results, and
extracted information.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns  # Optional dependency
from pathlib import Path
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RadarDataVisualizer:
    """
    Visualizes radar data and processing results.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Path to output directory containing processed data
        """
        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / 'raw_sim'
        self.rds_dir = self.output_dir / 'rds'
        self.angles_dir = self.output_dir / 'angles'
        self.velocities_dir = self.output_dir / 'velocities'
        self.poses_dir = self.output_dir / 'poses'
        
        logger.info(f"Initialized visualizer for: {output_dir}")
    
    def visualize_raw_signals(self, frame_idx: int = 0, antenna_idx: int = 0):
        """
        Visualize raw FMCW signals.
        
        Args:
            frame_idx: Frame index to visualize
            antenna_idx: Antenna index to visualize
        """
        frame_file = self.raw_dir / f"frame_{frame_idx:04d}.npy"
        
        if not frame_file.exists():
            logger.error(f"Raw signal file not found: {frame_file}")
            return
        
        # Load raw signals
        raw_signals = np.load(frame_file)
        logger.info(f"Raw signals shape: {raw_signals.shape}")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Time domain signal (real part)
        ax = axes[0, 0]
        signal_real = np.real(raw_signals[antenna_idx, 0, :])  # First chirp, real part
        time_axis = np.linspace(0, 40e-6, len(signal_real))  # 40 μs chirp duration
        ax.plot(time_axis * 1e6, signal_real)
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Raw Signal (Real) - Antenna {antenna_idx}, Chirp 0')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Time domain signal (imaginary part)
        ax = axes[0, 1]
        signal_imag = np.imag(raw_signals[antenna_idx, 0, :])  # First chirp, imaginary part
        ax.plot(time_axis * 1e6, signal_imag)
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Raw Signal (Imag) - Antenna {antenna_idx}, Chirp 0')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Magnitude spectrum
        ax = axes[1, 0]
        signal_mag = np.abs(raw_signals[antenna_idx, 0, :])
        ax.plot(time_axis * 1e6, signal_mag)
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel('Magnitude')
        ax.set_title(f'Signal Magnitude - Antenna {antenna_idx}, Chirp 0')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Phase
        ax = axes[1, 1]
        signal_phase = np.angle(raw_signals[antenna_idx, 0, :])
        ax.plot(time_axis * 1e6, signal_phase)
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel('Phase (rad)')
        ax.set_title(f'Signal Phase - Antenna {antenna_idx}, Chirp 0')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'raw_signals_frame_{frame_idx:04d}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Raw signals visualization saved for frame {frame_idx}")
    
    def visualize_rds(self, frame_idx: int = 0, antenna_idx: int = 0):
        """
        Visualize Range-Doppler Spectrum.
        
        Args:
            frame_idx: Frame index to visualize
            antenna_idx: Antenna index to visualize
        """
        rds_file = self.rds_dir / f"frame_{frame_idx:04d}_rds.npy"
        peaks_file = self.rds_dir / f"frame_{frame_idx:04d}_peaks.npz"
        
        if not rds_file.exists():
            logger.error(f"RDS file not found: {rds_file}")
            return
        
        # Load RDS data
        rds = np.load(rds_file)
        logger.info(f"RDS shape: {rds.shape}")
        
        # Load peaks if available
        peaks_info = None
        if peaks_file.exists():
            peaks_data = np.load(peaks_file, allow_pickle=True)
            peaks_info = dict(peaks_data)
            logger.info(f"Found {len(peaks_info['peaks'])} peaks")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: RDS magnitude
        ax = axes[0, 0]
        rds_mag = np.abs(rds[antenna_idx, :, :])
        rds_db = 10 * np.log10(rds_mag + 1e-12)
        
        im = ax.imshow(rds_db, aspect='auto', origin='lower', cmap='jet')
        ax.set_xlabel('Doppler Bin')
        ax.set_ylabel('Range Bin')
        ax.set_title(f'Range-Doppler Spectrum - Antenna {antenna_idx}')
        plt.colorbar(im, ax=ax, label='Power (dB)')
        
        # Plot 2: RDS phase
        ax = axes[0, 1]
        rds_phase = np.angle(rds[antenna_idx, :, :])
        im = ax.imshow(rds_phase, aspect='auto', origin='lower', cmap='hsv')
        ax.set_xlabel('Doppler Bin')
        ax.set_ylabel('Range Bin')
        ax.set_title(f'RDS Phase - Antenna {antenna_idx}')
        plt.colorbar(im, ax=ax, label='Phase (rad)')
        
        # Plot 3: Range profile (sum over Doppler)
        ax = axes[1, 0]
        range_profile = np.sum(rds_mag, axis=1)
        range_bins = np.arange(len(range_profile))
        ax.plot(range_bins, 10 * np.log10(range_profile + 1e-12))
        ax.set_xlabel('Range Bin')
        ax.set_ylabel('Power (dB)')
        ax.set_title('Range Profile')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Doppler profile (sum over range)
        ax = axes[1, 1]
        doppler_profile = np.sum(rds_mag, axis=0)
        doppler_bins = np.arange(len(doppler_profile))
        ax.plot(doppler_bins, 10 * np.log10(doppler_profile + 1e-12))
        ax.set_xlabel('Doppler Bin')
        ax.set_ylabel('Power (dB)')
        ax.set_title('Doppler Profile')
        ax.grid(True, alpha=0.3)
        
        # Overlay peaks if available
        if peaks_info and len(peaks_info['peaks']) > 0:
            for peak in peaks_info['peaks']:
                if peak['antenna'] == antenna_idx:
                    ax.scatter(peak['doppler_bin'], peak['range_bin'], 
                             c='red', s=50, marker='x', linewidth=2)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'rds_frame_{frame_idx:04d}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        logger.info(f"RDS visualization saved for frame {frame_idx}")
    
    def visualize_peaks(self, frame_idx: int = 0):
        """
        Visualize detected peaks.
        
        Args:
            frame_idx: Frame index to visualize
        """
        peaks_file = self.rds_dir / f"frame_{frame_idx:04d}_peaks.npz"
        
        if not peaks_file.exists():
            logger.error(f"Peaks file not found: {peaks_file}")
            return
        
        # Load peaks data
        peaks_data = np.load(peaks_file, allow_pickle=True)
        peaks_info = dict(peaks_data)
        
        if len(peaks_info['peaks']) == 0:
            logger.warning(f"No peaks found in frame {frame_idx}")
            return
        
        logger.info(f"Visualizing {len(peaks_info['peaks'])} peaks")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Range vs Doppler scatter plot
        ax = axes[0, 0]
        ranges = [peak['range_m'] for peak in peaks_info['peaks']]
        dopplers = [peak['doppler_hz'] for peak in peaks_info['peaks']]
        powers = [peak['power_db'] for peak in peaks_info['peaks']]
        
        scatter = ax.scatter(dopplers, ranges, c=powers, cmap='viridis', s=50, alpha=0.7)
        ax.set_xlabel('Doppler Frequency (Hz)')
        ax.set_ylabel('Range (m)')
        ax.set_title('Detected Peaks: Range vs Doppler')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Power (dB)')
        
        # Plot 2: Range distribution
        ax = axes[0, 1]
        ax.hist(ranges, bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Range (m)')
        ax.set_ylabel('Count')
        ax.set_title('Range Distribution')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Doppler distribution
        ax = axes[1, 0]
        ax.hist(dopplers, bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Doppler Frequency (Hz)')
        ax.set_ylabel('Count')
        ax.set_title('Doppler Distribution')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Power distribution
        ax = axes[1, 1]
        ax.hist(powers, bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Power (dB)')
        ax.set_ylabel('Count')
        ax.set_title('Power Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'peaks_frame_{frame_idx:04d}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Peaks visualization saved for frame {frame_idx}")
    
    def visualize_angles(self, frame_idx: int = 0):
        """
        Visualize angle estimates.
        
        Args:
            frame_idx: Frame index to visualize
        """
        angles_file = self.angles_dir / f"frame_{frame_idx:04d}_angles.npz"
        
        if not angles_file.exists():
            logger.warning(f"Angles file not found: {angles_file}")
            return
        
        # Load angles data
        angles_data = np.load(angles_file, allow_pickle=True)
        targets = angles_data['targets'].item()
        
        if len(targets) == 0:
            logger.warning(f"No angle estimates found in frame {frame_idx}")
            return
        
        logger.info(f"Visualizing {len(targets)} angle estimates")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Angle distribution
        ax = axes[0, 0]
        angles = [target['azimuth_deg'] for target in targets]
        ax.hist(angles, bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Azimuth Angle (degrees)')
        ax.set_ylabel('Count')
        ax.set_title('Angle Distribution')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Range vs Angle
        ax = axes[0, 1]
        ranges = [target['range_m'] for target in targets]
        powers = [target['power_db'] for target in targets]
        
        scatter = ax.scatter(angles, ranges, c=powers, cmap='viridis', s=50, alpha=0.7)
        ax.set_xlabel('Azimuth Angle (degrees)')
        ax.set_ylabel('Range (m)')
        ax.set_title('Range vs Angle')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Power (dB)')
        
        # Plot 3: Polar plot of angles
        ax = axes[1, 0]
        ax = plt.subplot(2, 2, 3, projection='polar')
        angles_rad = np.radians(angles)
        ax.scatter(angles_rad, ranges, c=powers, cmap='viridis', s=50, alpha=0.7)
        ax.set_title('Polar Plot: Angles and Ranges')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        
        # Plot 4: Power vs Angle
        ax = axes[1, 1]
        ax.scatter(angles, powers, alpha=0.7)
        ax.set_xlabel('Azimuth Angle (degrees)')
        ax.set_ylabel('Power (dB)')
        ax.set_title('Power vs Angle')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'angles_frame_{frame_idx:04d}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Angles visualization saved for frame {frame_idx}")
    
    def visualize_velocities(self):
        """
        Visualize velocity estimates.
        """
        velocity_files = list(self.velocities_dir.glob('*_velocity.npz'))
        
        if not velocity_files:
            logger.warning("No velocity files found")
            return
        
        logger.info(f"Found {len(velocity_files)} velocity files")
        
        # Collect velocity data
        velocities = []
        angular_velocities = []
        timestamps = []
        
        for i, velocity_file in enumerate(sorted(velocity_files)):
            try:
                velocity_data = np.load(velocity_file, allow_pickle=True)
                if velocity_data['success']:
                    velocities.append(velocity_data['velocity'])
                    angular_velocities.append(velocity_data['angular_velocity'])
                    timestamps.append(i * 0.1)  # Assume 0.1s time step
            except Exception as e:
                logger.warning(f"Error loading {velocity_file}: {e}")
                continue
        
        if not velocities:
            logger.warning("No valid velocity data found")
            return
        
        velocities = np.array(velocities)
        angular_velocities = np.array(angular_velocities)
        timestamps = np.array(timestamps)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Translational velocities
        ax = axes[0, 0]
        ax.plot(timestamps, velocities[:, 0], label='vx', linewidth=2)
        ax.plot(timestamps, velocities[:, 1], label='vy', linewidth=2)
        ax.plot(timestamps, velocities[:, 2], label='vz', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title('Translational Velocities')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Angular velocities
        ax = axes[0, 1]
        ax.plot(timestamps, angular_velocities[:, 0], label='wx', linewidth=2)
        ax.plot(timestamps, angular_velocities[:, 1], label='wy', linewidth=2)
        ax.plot(timestamps, angular_velocities[:, 2], label='wz', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angular Velocity (rad/s)')
        ax.set_title('Angular Velocities')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Speed
        ax = axes[1, 0]
        speed = np.linalg.norm(velocities, axis=1)
        ax.plot(timestamps, speed, linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (m/s)')
        ax.set_title('Vehicle Speed')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Angular speed
        ax = axes[1, 1]
        angular_speed = np.linalg.norm(angular_velocities, axis=1)
        ax.plot(timestamps, angular_speed, linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angular Speed (rad/s)')
        ax.set_title('Angular Speed')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'velocities.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        logger.info("Velocity visualization saved")
    
    def visualize_trajectory(self):
        """
        Visualize pose trajectory.
        """
        trajectory_file = self.poses_dir / 'trajectory.npz'
        
        if not trajectory_file.exists():
            logger.warning("No trajectory file found")
            return
        
        # Load trajectory data
        trajectory_data = np.load(trajectory_file, allow_pickle=True)
        
        positions = trajectory_data['positions']
        orientations = trajectory_data['orientations']
        timestamps = trajectory_data['timestamps']
        
        logger.info(f"Visualizing trajectory with {len(positions)} points")
        
        # Create visualization
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
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'trajectory.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        logger.info("Trajectory visualization saved")
    
    def generate_summary_report(self):
        """
        Generate a summary report of all processing results.
        """
        logger.info("Generating summary report...")
        
        # Count files in each directory
        raw_files = len(list(self.raw_dir.glob('frame_*.npy')))
        rds_files = len(list(self.rds_dir.glob('*_rds.npy')))
        peak_files = len(list(self.rds_dir.glob('*_peaks.npz')))
        angle_files = len(list(self.angles_dir.glob('*_angles.npz')))
        velocity_files = len(list(self.velocities_dir.glob('*_velocity.npz')))
        
        # Create summary plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Processing pipeline status
        ax = axes[0, 0]
        stages = ['Raw Signals', 'RDS', 'Peaks', 'Angles', 'Velocities']
        counts = [raw_files, rds_files, peak_files, angle_files, velocity_files]
        colors = ['green' if c > 0 else 'red' for c in counts]
        
        bars = ax.bar(stages, counts, color=colors, alpha=0.7)
        ax.set_ylabel('Number of Files')
        ax.set_title('Processing Pipeline Status')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count}', ha='center', va='bottom')
        
        # Plot 2: File size distribution
        ax = axes[0, 1]
        file_sizes = []
        file_types = []
        
        for stage, count in zip(stages, counts):
            if count > 0:
                # Get file sizes
                if stage == 'Raw Signals':
                    files = list(self.raw_dir.glob('frame_*.npy'))
                elif stage == 'RDS':
                    files = list(self.rds_dir.glob('*_rds.npy'))
                elif stage == 'Peaks':
                    files = list(self.rds_dir.glob('*_peaks.npz'))
                elif stage == 'Angles':
                    files = list(self.angles_dir.glob('*_angles.npz'))
                elif stage == 'Velocities':
                    files = list(self.velocities_dir.glob('*_velocity.npz'))
                
                sizes = [f.stat().st_size / 1024 / 1024 for f in files[:5]]  # First 5 files
                file_sizes.extend(sizes)
                file_types.extend([stage] * len(sizes))
        
        if file_sizes:
            # Simple bar plot instead of seaborn
            unique_types = list(set(file_types))
            type_sizes = [np.mean([s for s, t in zip(file_sizes, file_types) if t == typ]) for typ in unique_types]
            ax.bar(unique_types, type_sizes, alpha=0.7)
            ax.set_title('File Size Distribution')
            ax.set_ylabel('Size (MB)')
            ax.tick_params(axis='x', rotation=45)
        
        # Plot 3: Processing timeline
        ax = axes[1, 0]
        if raw_files > 0:
            frame_indices = list(range(raw_files))
            ax.plot(frame_indices, [1] * raw_files, 'o-', label='Raw Signals', linewidth=2)
        if rds_files > 0:
            ax.plot(frame_indices[:rds_files], [2] * rds_files, 'o-', label='RDS', linewidth=2)
        if peak_files > 0:
            ax.plot(frame_indices[:peak_files], [3] * peak_files, 'o-', label='Peaks', linewidth=2)
        if angle_files > 0:
            ax.plot(frame_indices[:angle_files], [4] * angle_files, 'o-', label='Angles', linewidth=2)
        if velocity_files > 0:
            ax.plot(frame_indices[:velocity_files], [5] * velocity_files, 'o-', label='Velocities', linewidth=2)
        
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Processing Stage')
        ax.set_title('Processing Timeline')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = f"""
        Processing Summary:
        
        Raw Signals: {raw_files} files
        RDS: {rds_files} files  
        Peaks: {peak_files} files
        Angles: {angle_files} files
        Velocities: {velocity_files} files
        
        Total Processing Stages: {sum(1 for c in counts if c > 0)}/5
        
        Status: {'✅ Complete' if all(c > 0 for c in counts) else '⚠️ Partial'}
        """
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'summary_report.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        logger.info("Summary report generated")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Visualize radar processing results')
    parser.add_argument('--output', required=True, help='Output directory containing processed data')
    parser.add_argument('--frame', type=int, default=0, help='Frame index to visualize')
    parser.add_argument('--antenna', type=int, default=0, help='Antenna index to visualize')
    parser.add_argument('--all', action='store_true', help='Generate all visualizations')
    parser.add_argument('--summary', action='store_true', help='Generate summary report')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = RadarDataVisualizer(args.output)
    
    if args.all:
        logger.info("Generating all visualizations...")
        visualizer.visualize_raw_signals(args.frame, args.antenna)
        visualizer.visualize_rds(args.frame, args.antenna)
        visualizer.visualize_peaks(args.frame)
        visualizer.visualize_angles(args.frame)
        visualizer.visualize_velocities()
        visualizer.visualize_trajectory()
        visualizer.generate_summary_report()
    else:
        # Generate specific visualizations
        visualizer.visualize_raw_signals(args.frame, args.antenna)
        visualizer.visualize_rds(args.frame, args.antenna)
        visualizer.visualize_peaks(args.frame)
        
        if args.summary:
            visualizer.generate_summary_report()
    
    logger.info("Visualization complete!")


if __name__ == "__main__":
    main()
