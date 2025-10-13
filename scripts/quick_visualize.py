#!/usr/bin/env python3
"""
Quick Visualization Script

Simple visualization of radar processing results to understand the data.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def visualize_peaks(output_dir: str, frame_idx: int = 0):
    """
    Visualize detected peaks from a single frame.
    """
    output_path = Path(output_dir)
    rds_dir = output_path / 'rds'
    peaks_file = rds_dir / f"frame_{frame_idx:04d}_peaks.npz"
    
    if not peaks_file.exists():
        logger.error(f"Peaks file not found: {peaks_file}")
        return
    
    # Load peaks data
    peaks_data = np.load(peaks_file, allow_pickle=True)
    peaks_info = dict(peaks_data)
    
    if len(peaks_info['peaks']) == 0:
        logger.warning(f"No peaks found in frame {frame_idx}")
        return
    
    peaks = peaks_info['peaks']
    logger.info(f"Visualizing {len(peaks)} peaks from frame {frame_idx}")
    
    # Extract data
    ranges = np.array([peak['range_m'] for peak in peaks])
    dopplers = np.array([peak['doppler_hz'] for peak in peaks])
    powers = np.array([peak['power_db'] for peak in peaks])
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Range vs Doppler scatter plot
    ax = axes[0, 0]
    scatter = ax.scatter(dopplers, ranges, c=powers, cmap='viridis', s=20, alpha=0.6)
    ax.set_xlabel('Doppler Frequency (Hz)')
    ax.set_ylabel('Range (m)')
    ax.set_title(f'Detected Peaks: Range vs Doppler (Frame {frame_idx})')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Power (dB)')
    
    # Plot 2: Range distribution
    ax = axes[0, 1]
    ax.hist(ranges, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Count')
    ax.set_title('Range Distribution')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Doppler distribution
    ax = axes[1, 0]
    ax.hist(dopplers, bins=30, alpha=0.7, edgecolor='black', color='lightcoral')
    ax.set_xlabel('Doppler Frequency (Hz)')
    ax.set_ylabel('Count')
    ax.set_title('Doppler Distribution')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Power distribution
    ax = axes[1, 1]
    ax.hist(powers, bins=30, alpha=0.7, edgecolor='black', color='lightgreen')
    ax.set_xlabel('Power (dB)')
    ax.set_ylabel('Count')
    ax.set_title('Power Distribution')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / f'peaks_analysis_frame_{frame_idx:04d}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\n=== Peak Analysis for Frame {frame_idx} ===")
    print(f"Total peaks: {len(peaks)}")
    print(f"Range: {ranges.min():.1f} - {ranges.max():.1f} m (mean: {ranges.mean():.1f} m)")
    print(f"Doppler: {dopplers.min():.1f} - {dopplers.max():.1f} Hz (mean: {dopplers.mean():.1f} Hz)")
    print(f"Power: {powers.min():.1f} - {powers.max():.1f} dB (mean: {powers.mean():.1f} dB)")
    
    # Target classification
    close_targets = np.sum(ranges < 20)
    medium_targets = np.sum((ranges >= 20) & (ranges < 50))
    far_targets = np.sum(ranges >= 50)
    moving_targets = np.sum(np.abs(dopplers) > 1)
    stationary_targets = np.sum(np.abs(dopplers) <= 1)
    
    print(f"\nTarget Classification:")
    print(f"  Close targets (<20m): {close_targets} ({close_targets/len(peaks)*100:.1f}%)")
    print(f"  Medium targets (20-50m): {medium_targets} ({medium_targets/len(peaks)*100:.1f}%)")
    print(f"  Far targets (>50m): {far_targets} ({far_targets/len(peaks)*100:.1f}%)")
    print(f"  Moving targets: {moving_targets} ({moving_targets/len(peaks)*100:.1f}%)")
    print(f"  Stationary targets: {stationary_targets} ({stationary_targets/len(peaks)*100:.1f}%)")

def visualize_rds(output_dir: str, frame_idx: int = 0, antenna_idx: int = 0):
    """
    Visualize Range-Doppler Spectrum.
    """
    output_path = Path(output_dir)
    rds_dir = output_path / 'rds'
    rds_file = rds_dir / f"frame_{frame_idx:04d}_rds.npy"
    
    if not rds_file.exists():
        logger.error(f"RDS file not found: {rds_file}")
        return
    
    # Load RDS data
    rds = np.load(rds_file)
    logger.info(f"RDS shape: {rds.shape}")
    
    # Extract RDS for specific antenna
    rds_antenna = rds[antenna_idx, :, :]
    rds_mag = np.abs(rds_antenna)
    rds_db = 10 * np.log10(rds_mag + 1e-12)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: RDS magnitude
    ax = axes[0, 0]
    im = ax.imshow(rds_db, aspect='auto', origin='lower', cmap='jet')
    ax.set_xlabel('Doppler Bin')
    ax.set_ylabel('Range Bin')
    ax.set_title(f'Range-Doppler Spectrum - Antenna {antenna_idx}')
    plt.colorbar(im, ax=ax, label='Power (dB)')
    
    # Plot 2: Range profile
    ax = axes[0, 1]
    range_profile = np.sum(rds_mag, axis=1)
    ax.plot(range_profile)
    ax.set_xlabel('Range Bin')
    ax.set_ylabel('Power')
    ax.set_title('Range Profile')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Doppler profile
    ax = axes[1, 0]
    doppler_profile = np.sum(rds_mag, axis=0)
    ax.plot(doppler_profile)
    ax.set_xlabel('Doppler Bin')
    ax.set_ylabel('Power')
    ax.set_title('Doppler Profile')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: RDS phase
    ax = axes[1, 1]
    rds_phase = np.angle(rds_antenna)
    im = ax.imshow(rds_phase, aspect='auto', origin='lower', cmap='hsv')
    ax.set_xlabel('Doppler Bin')
    ax.set_ylabel('Range Bin')
    ax.set_title(f'RDS Phase - Antenna {antenna_idx}')
    plt.colorbar(im, ax=ax, label='Phase (rad)')
    
    plt.tight_layout()
    plt.savefig(output_path / f'rds_analysis_frame_{frame_idx:04d}_antenna_{antenna_idx}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print RDS statistics
    print(f"\n=== RDS Analysis for Frame {frame_idx}, Antenna {antenna_idx} ===")
    print(f"RDS shape: {rds.shape}")
    print(f"Max power: {rds_db.max():.1f} dB")
    print(f"Min power: {rds_db.min():.1f} dB")
    print(f"Mean power: {rds_db.mean():.1f} dB")
    print(f"Dynamic range: {rds_db.max() - rds_db.min():.1f} dB")

def compare_frames(output_dir: str):
    """
    Compare multiple frames to see temporal changes.
    """
    output_path = Path(output_dir)
    rds_dir = output_path / 'rds'
    
    # Get all available frames
    peak_files = list(rds_dir.glob('*_peaks.npz'))
    frame_indices = [int(f.stem.split('_')[1]) for f in peak_files]
    frame_indices.sort()
    
    if len(frame_indices) < 2:
        logger.warning("Need at least 2 frames for comparison")
        return
    
    logger.info(f"Comparing {len(frame_indices)} frames")
    
    # Collect data from all frames
    frame_data = []
    
    for frame_idx in frame_indices:
        peaks_file = rds_dir / f"frame_{frame_idx:04d}_peaks.npz"
        
        if not peaks_file.exists():
            continue
        
        peaks_data = np.load(peaks_file, allow_pickle=True)
        peaks_info = dict(peaks_data)
        
        if len(peaks_info['peaks']) == 0:
            continue
        
        peaks = peaks_info['peaks']
        ranges = np.array([peak['range_m'] for peak in peaks])
        dopplers = np.array([peak['doppler_hz'] for peak in peaks])
        powers = np.array([peak['power_db'] for peak in peaks])
        
        frame_data.append({
            'frame_idx': frame_idx,
            'num_peaks': len(peaks),
            'mean_range': np.mean(ranges),
            'mean_doppler': np.mean(dopplers),
            'mean_power': np.mean(powers),
            'close_targets': np.sum(ranges < 20),
            'moving_targets': np.sum(np.abs(dopplers) > 1)
        })
    
    if not frame_data:
        logger.warning("No frame data found for comparison")
        return
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    frame_indices = [d['frame_idx'] for d in frame_data]
    num_peaks = [d['num_peaks'] for d in frame_data]
    mean_ranges = [d['mean_range'] for d in frame_data]
    mean_dopplers = [d['mean_doppler'] for d in frame_data]
    close_targets = [d['close_targets'] for d in frame_data]
    moving_targets = [d['moving_targets'] for d in frame_data]
    
    # Plot 1: Peak count over time
    ax = axes[0, 0]
    ax.plot(frame_indices, num_peaks, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Number of Peaks')
    ax.set_title('Peak Count Over Time')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Mean range over time
    ax = axes[0, 1]
    ax.plot(frame_indices, mean_ranges, 'o-', linewidth=2, markersize=6, color='green')
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Mean Range (m)')
    ax.set_title('Mean Range Over Time')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Mean Doppler over time
    ax = axes[1, 0]
    ax.plot(frame_indices, mean_dopplers, 'o-', linewidth=2, markersize=6, color='red')
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Mean Doppler (Hz)')
    ax.set_title('Mean Doppler Over Time')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Target classification over time
    ax = axes[1, 1]
    ax.plot(frame_indices, close_targets, 'o-', label='Close Targets', linewidth=2, markersize=6)
    ax.plot(frame_indices, moving_targets, 'o-', label='Moving Targets', linewidth=2, markersize=6)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Number of Targets')
    ax.set_title('Target Classification Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'frame_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print comparison summary
    print(f"\n=== Frame Comparison Summary ===")
    print(f"Frames analyzed: {len(frame_data)}")
    print(f"Average peaks per frame: {np.mean(num_peaks):.1f}")
    print(f"Peak count range: {np.min(num_peaks)} - {np.max(num_peaks)}")
    print(f"Mean range: {np.mean(mean_ranges):.1f} m")
    print(f"Mean Doppler: {np.mean(mean_dopplers):.1f} Hz")

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Quick visualization of radar data')
    parser.add_argument('--output', required=True, help='Output directory containing processed data')
    parser.add_argument('--frame', type=int, default=0, help='Frame index to visualize')
    parser.add_argument('--antenna', type=int, default=0, help='Antenna index to visualize')
    parser.add_argument('--peaks', action='store_true', help='Visualize peaks')
    parser.add_argument('--rds', action='store_true', help='Visualize RDS')
    parser.add_argument('--compare', action='store_true', help='Compare frames')
    parser.add_argument('--all', action='store_true', help='Run all visualizations')
    
    args = parser.parse_args()
    
    if args.all:
        visualize_peaks(args.output, args.frame)
        visualize_rds(args.output, args.frame, args.antenna)
        compare_frames(args.output)
    elif args.peaks:
        visualize_peaks(args.output, args.frame)
    elif args.rds:
        visualize_rds(args.output, args.frame, args.antenna)
    elif args.compare:
        compare_frames(args.output)
    else:
        # Default: show peaks
        visualize_peaks(args.output, args.frame)
    
    logger.info("Visualization complete!")

if __name__ == "__main__":
    main()
