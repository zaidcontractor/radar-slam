#!/usr/bin/env python3
"""
Data Analysis Script

This script analyzes the processed radar data to extract meaningful insights
about the radar scene, detected targets, and processing quality.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RadarDataAnalyzer:
    """
    Analyzes radar data to extract meaningful insights.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize analyzer.
        
        Args:
            output_dir: Path to output directory containing processed data
        """
        self.output_dir = Path(output_dir)
        self.rds_dir = self.output_dir / 'rds'
        
        logger.info(f"Initialized analyzer for: {output_dir}")
    
    def analyze_peaks(self, frame_idx: int = 0):
        """
        Analyze detected peaks to understand the radar scene.
        
        Args:
            frame_idx: Frame index to analyze
        """
        peaks_file = self.rds_dir / f"frame_{frame_idx:04d}_peaks.npz"
        
        if not peaks_file.exists():
            logger.error(f"Peaks file not found: {peaks_file}")
            return None
        
        # Load peaks data
        peaks_data = np.load(peaks_file, allow_pickle=True)
        peaks_info = dict(peaks_data)
        
        if len(peaks_info['peaks']) == 0:
            logger.warning(f"No peaks found in frame {frame_idx}")
            return None
        
        peaks = peaks_info['peaks']
        logger.info(f"Analyzing {len(peaks)} peaks from frame {frame_idx}")
        
        # Extract data
        ranges = np.array([peak['range_m'] for peak in peaks])
        dopplers = np.array([peak['doppler_hz'] for peak in peaks])
        powers = np.array([peak['power_db'] for peak in peaks])
        antennas = np.array([peak['antenna'] for peak in peaks])
        
        # Basic statistics
        analysis = {
            'num_peaks': len(peaks),
            'range_stats': {
                'min': np.min(ranges),
                'max': np.max(ranges),
                'mean': np.mean(ranges),
                'std': np.std(ranges),
                'median': np.median(ranges)
            },
            'doppler_stats': {
                'min': np.min(dopplers),
                'max': np.max(dopplers),
                'mean': np.mean(dopplers),
                'std': np.std(dopplers),
                'median': np.median(dopplers)
            },
            'power_stats': {
                'min': np.min(powers),
                'max': np.max(powers),
                'mean': np.mean(powers),
                'std': np.std(powers),
                'median': np.median(powers)
            },
            'antenna_distribution': {
                'unique_antennas': len(np.unique(antennas)),
                'antenna_counts': {ant: np.sum(antennas == ant) for ant in np.unique(antennas)}
            }
        }
        
        # Target classification based on range and Doppler
        close_targets = np.sum(ranges < 20)  # Within 20m
        medium_targets = np.sum((ranges >= 20) & (ranges < 50))  # 20-50m
        far_targets = np.sum(ranges >= 50)  # Beyond 50m
        
        moving_targets = np.sum(np.abs(dopplers) > 1)  # Moving (|doppler| > 1 Hz)
        stationary_targets = np.sum(np.abs(dopplers) <= 1)  # Stationary
        
        analysis['target_classification'] = {
            'close_targets': close_targets,
            'medium_targets': medium_targets,
            'far_targets': far_targets,
            'moving_targets': moving_targets,
            'stationary_targets': stationary_targets
        }
        
        # Print analysis results
        print(f"\n=== Peak Analysis for Frame {frame_idx} ===")
        print(f"Total peaks detected: {analysis['num_peaks']}")
        print(f"\nRange Analysis:")
        print(f"  Min range: {analysis['range_stats']['min']:.2f} m")
        print(f"  Max range: {analysis['range_stats']['max']:.2f} m")
        print(f"  Mean range: {analysis['range_stats']['mean']:.2f} m")
        print(f"  Range std: {analysis['range_stats']['std']:.2f} m")
        
        print(f"\nDoppler Analysis:")
        print(f"  Min Doppler: {analysis['doppler_stats']['min']:.2f} Hz")
        print(f"  Max Doppler: {analysis['doppler_stats']['max']:.2f} Hz")
        print(f"  Mean Doppler: {analysis['doppler_stats']['mean']:.2f} Hz")
        print(f"  Doppler std: {analysis['doppler_stats']['std']:.2f} Hz")
        
        print(f"\nPower Analysis:")
        print(f"  Min power: {analysis['power_stats']['min']:.2f} dB")
        print(f"  Max power: {analysis['power_stats']['max']:.2f} dB")
        print(f"  Mean power: {analysis['power_stats']['mean']:.2f} dB")
        
        print(f"\nTarget Classification:")
        print(f"  Close targets (<20m): {close_targets}")
        print(f"  Medium targets (20-50m): {medium_targets}")
        print(f"  Far targets (>50m): {far_targets}")
        print(f"  Moving targets: {moving_targets}")
        print(f"  Stationary targets: {stationary_targets}")
        
        print(f"\nAntenna Distribution:")
        for ant, count in analysis['antenna_distribution']['antenna_counts'].items():
            print(f"  Antenna {ant}: {count} peaks")
        
        return analysis
    
    def analyze_rds_quality(self, frame_idx: int = 0, antenna_idx: int = 0):
        """
        Analyze RDS quality and signal characteristics.
        
        Args:
            frame_idx: Frame index to analyze
            antenna_idx: Antenna index to analyze
        """
        rds_file = self.rds_dir / f"frame_{frame_idx:04d}_rds.npy"
        
        if not rds_file.exists():
            logger.error(f"RDS file not found: {rds_file}")
            return None
        
        # Load RDS data
        rds = np.load(rds_file)
        logger.info(f"Analyzing RDS shape: {rds.shape}")
        
        # Extract RDS for specific antenna
        rds_antenna = rds[antenna_idx, :, :]
        rds_mag = np.abs(rds_antenna)
        rds_db = 10 * np.log10(rds_mag + 1e-12)
        
        # Calculate statistics
        analysis = {
            'rds_shape': rds.shape,
            'antenna_idx': antenna_idx,
            'signal_characteristics': {
                'max_power': np.max(rds_db),
                'min_power': np.min(rds_db),
                'mean_power': np.mean(rds_db),
                'std_power': np.std(rds_db),
                'dynamic_range': np.max(rds_db) - np.min(rds_db)
            },
            'noise_floor': np.percentile(rds_db, 10),  # Bottom 10% as noise floor
            'signal_peaks': np.sum(rds_db > np.percentile(rds_db, 90)),  # Top 10% as peaks
            'snr_estimate': np.max(rds_db) - np.percentile(rds_db, 10)
        }
        
        # Range and Doppler profiles
        range_profile = np.sum(rds_mag, axis=1)
        doppler_profile = np.sum(rds_mag, axis=0)
        
        analysis['range_profile'] = {
            'max_range_bin': np.argmax(range_profile),
            'range_energy': np.sum(range_profile),
            'range_peaks': np.sum(range_profile > np.percentile(range_profile, 90))
        }
        
        analysis['doppler_profile'] = {
            'max_doppler_bin': np.argmax(doppler_profile),
            'doppler_energy': np.sum(doppler_profile),
            'doppler_peaks': np.sum(doppler_profile > np.percentile(doppler_profile, 90))
        }
        
        # Print analysis results
        print(f"\n=== RDS Quality Analysis for Frame {frame_idx}, Antenna {antenna_idx} ===")
        print(f"RDS shape: {analysis['rds_shape']}")
        print(f"\nSignal Characteristics:")
        print(f"  Max power: {analysis['signal_characteristics']['max_power']:.2f} dB")
        print(f"  Min power: {analysis['signal_characteristics']['min_power']:.2f} dB")
        print(f"  Mean power: {analysis['signal_characteristics']['mean_power']:.2f} dB")
        print(f"  Dynamic range: {analysis['signal_characteristics']['dynamic_range']:.2f} dB")
        print(f"  SNR estimate: {analysis['snr_estimate']:.2f} dB")
        
        print(f"\nRange Profile:")
        print(f"  Max range bin: {analysis['range_profile']['max_range_bin']}")
        print(f"  Range energy: {analysis['range_profile']['range_energy']:.2f}")
        print(f"  Range peaks: {analysis['range_profile']['range_peaks']}")
        
        print(f"\nDoppler Profile:")
        print(f"  Max Doppler bin: {analysis['doppler_profile']['max_doppler_bin']}")
        print(f"  Doppler energy: {analysis['doppler_profile']['doppler_energy']:.2f}")
        print(f"  Doppler peaks: {analysis['doppler_profile']['doppler_peaks']}")
        
        return analysis
    
    def compare_frames(self, frame_indices: list = None):
        """
        Compare multiple frames to understand temporal changes.
        
        Args:
            frame_indices: List of frame indices to compare
        """
        if frame_indices is None:
            # Get all available frames
            peak_files = list(self.rds_dir.glob('*_peaks.npz'))
            frame_indices = [int(f.stem.split('_')[1]) for f in peak_files]
            frame_indices.sort()
        
        logger.info(f"Comparing {len(frame_indices)} frames")
        
        # Collect data from all frames
        frame_data = []
        
        for frame_idx in frame_indices:
            peaks_file = self.rds_dir / f"frame_{frame_idx:04d}_peaks.npz"
            
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
                'std_range': np.std(ranges),
                'mean_doppler': np.mean(dopplers),
                'std_doppler': np.std(dopplers),
                'mean_power': np.mean(powers),
                'max_power': np.max(powers),
                'close_targets': np.sum(ranges < 20),
                'moving_targets': np.sum(np.abs(dopplers) > 1)
            })
        
        if not frame_data:
            logger.warning("No frame data found for comparison")
            return None
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(frame_data)
        
        # Print comparison results
        print(f"\n=== Frame Comparison Analysis ===")
        print(f"Frames analyzed: {len(frame_data)}")
        print(f"\nPeak Count Statistics:")
        print(f"  Mean peaks per frame: {df['num_peaks'].mean():.1f}")
        print(f"  Std peaks per frame: {df['num_peaks'].std():.1f}")
        print(f"  Min peaks: {df['num_peaks'].min()}")
        print(f"  Max peaks: {df['num_peaks'].max()}")
        
        print(f"\nRange Statistics:")
        print(f"  Mean range: {df['mean_range'].mean():.2f} m")
        print(f"  Range variation: {df['std_range'].mean():.2f} m")
        
        print(f"\nDoppler Statistics:")
        print(f"  Mean Doppler: {df['mean_doppler'].mean():.2f} Hz")
        print(f"  Doppler variation: {df['std_doppler'].mean():.2f} Hz")
        
        print(f"\nTarget Classification:")
        print(f"  Mean close targets: {df['close_targets'].mean():.1f}")
        print(f"  Mean moving targets: {df['moving_targets'].mean():.1f}")
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Peak count over time
        ax = axes[0, 0]
        ax.plot(df['frame_idx'], df['num_peaks'], 'o-', linewidth=2)
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Number of Peaks')
        ax.set_title('Peak Count Over Time')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Mean range over time
        ax = axes[0, 1]
        ax.plot(df['frame_idx'], df['mean_range'], 'o-', linewidth=2, color='green')
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Mean Range (m)')
        ax.set_title('Mean Range Over Time')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Mean Doppler over time
        ax = axes[1, 0]
        ax.plot(df['frame_idx'], df['mean_doppler'], 'o-', linewidth=2, color='red')
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Mean Doppler (Hz)')
        ax.set_title('Mean Doppler Over Time')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Target classification over time
        ax = axes[1, 1]
        ax.plot(df['frame_idx'], df['close_targets'], 'o-', label='Close Targets', linewidth=2)
        ax.plot(df['frame_idx'], df['moving_targets'], 'o-', label='Moving Targets', linewidth=2)
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Number of Targets')
        ax.set_title('Target Classification Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'frame_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return df
    
    def generate_insights_report(self):
        """
        Generate a comprehensive insights report.
        """
        logger.info("Generating insights report...")
        
        # Analyze all available frames
        peak_files = list(self.rds_dir.glob('*_peaks.npz'))
        
        if not peak_files:
            logger.warning("No peak files found for analysis")
            return
        
        logger.info(f"Analyzing {len(peak_files)} frames")
        
        # Collect overall statistics
        all_peaks = []
        all_ranges = []
        all_dopplers = []
        all_powers = []
        
        for peak_file in peak_files:
            try:
                peaks_data = np.load(peak_file, allow_pickle=True)
                peaks_info = dict(peaks_data)
                
                if len(peaks_info['peaks']) > 0:
                    peaks = peaks_info['peaks']
                    all_peaks.extend(peaks)
                    
                    ranges = [peak['range_m'] for peak in peaks]
                    dopplers = [peak['doppler_hz'] for peak in peaks]
                    powers = [peak['power_db'] for peak in peaks]
                    
                    all_ranges.extend(ranges)
                    all_dopplers.extend(dopplers)
                    all_powers.extend(powers)
            except Exception as e:
                logger.warning(f"Error processing {peak_file}: {e}")
                continue
        
        if not all_peaks:
            logger.warning("No peaks found in any frame")
            return
        
        # Calculate overall statistics
        all_ranges = np.array(all_ranges)
        all_dopplers = np.array(all_dopplers)
        all_powers = np.array(all_powers)
        
        # Generate insights
        insights = {
            'total_peaks': len(all_peaks),
            'total_frames': len(peak_files),
            'peaks_per_frame': len(all_peaks) / len(peak_files),
            'range_insights': {
                'min_range': np.min(all_ranges),
                'max_range': np.max(all_ranges),
                'mean_range': np.mean(all_ranges),
                'range_coverage': np.max(all_ranges) - np.min(all_ranges)
            },
            'doppler_insights': {
                'min_doppler': np.min(all_dopplers),
                'max_doppler': np.max(all_dopplers),
                'mean_doppler': np.mean(all_dopplers),
                'doppler_range': np.max(all_dopplers) - np.min(all_dopplers)
            },
            'power_insights': {
                'min_power': np.min(all_powers),
                'max_power': np.max(all_powers),
                'mean_power': np.mean(all_powers),
                'power_range': np.max(all_powers) - np.min(all_powers)
            }
        }
        
        # Target classification
        close_targets = np.sum(all_ranges < 20)
        medium_targets = np.sum((all_ranges >= 20) & (all_ranges < 50))
        far_targets = np.sum(all_ranges >= 50)
        moving_targets = np.sum(np.abs(all_dopplers) > 1)
        stationary_targets = np.sum(np.abs(all_dopplers) <= 1)
        
        insights['target_classification'] = {
            'close_targets': close_targets,
            'medium_targets': medium_targets,
            'far_targets': far_targets,
            'moving_targets': moving_targets,
            'stationary_targets': stationary_targets
        }
        
        # Print comprehensive report
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE RADAR DATA INSIGHTS REPORT")
        print(f"{'='*60}")
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"  Total peaks detected: {insights['total_peaks']}")
        print(f"  Total frames processed: {insights['total_frames']}")
        print(f"  Average peaks per frame: {insights['peaks_per_frame']:.1f}")
        
        print(f"\n📏 RANGE ANALYSIS:")
        print(f"  Range coverage: {insights['range_insights']['min_range']:.1f} - {insights['range_insights']['max_range']:.1f} m")
        print(f"  Mean detection range: {insights['range_insights']['mean_range']:.1f} m")
        print(f"  Range span: {insights['range_insights']['range_coverage']:.1f} m")
        
        print(f"\n🌊 DOPPLER ANALYSIS:")
        print(f"  Doppler range: {insights['doppler_insights']['min_doppler']:.1f} - {insights['doppler_insights']['max_doppler']:.1f} Hz")
        print(f"  Mean Doppler: {insights['doppler_insights']['mean_doppler']:.1f} Hz")
        print(f"  Doppler span: {insights['doppler_insights']['doppler_range']:.1f} Hz")
        
        print(f"\n⚡ POWER ANALYSIS:")
        print(f"  Power range: {insights['power_insights']['min_power']:.1f} - {insights['power_insights']['max_power']:.1f} dB")
        print(f"  Mean power: {insights['power_insights']['mean_power']:.1f} dB")
        print(f"  Dynamic range: {insights['power_insights']['power_range']:.1f} dB")
        
        print(f"\n🎯 TARGET CLASSIFICATION:")
        print(f"  Close targets (<20m): {close_targets} ({close_targets/len(all_peaks)*100:.1f}%)")
        print(f"  Medium targets (20-50m): {medium_targets} ({medium_targets/len(all_peaks)*100:.1f}%)")
        print(f"  Far targets (>50m): {far_targets} ({far_targets/len(all_peaks)*100:.1f}%)")
        print(f"  Moving targets: {moving_targets} ({moving_targets/len(all_peaks)*100:.1f}%)")
        print(f"  Stationary targets: {stationary_targets} ({stationary_targets/len(all_peaks)*100:.1f}%)")
        
        print(f"\n💡 INTERPRETATION:")
        if insights['range_insights']['mean_range'] < 30:
            print("  • Scene dominated by close-range targets")
        elif insights['range_insights']['mean_range'] > 100:
            print("  • Scene dominated by far-range targets")
        else:
            print("  • Balanced mix of near and far targets")
        
        if insights['doppler_insights']['mean_doppler'] > 5:
            print("  • Scene shows significant motion (positive Doppler)")
        elif insights['doppler_insights']['mean_doppler'] < -5:
            print("  • Scene shows significant motion (negative Doppler)")
        else:
            print("  • Scene shows relatively stationary targets")
        
        if moving_targets > stationary_targets:
            print("  • Dynamic scene with many moving objects")
        else:
            print("  • Static scene with mostly stationary objects")
        
        print(f"\n{'='*60}")
        
        return insights


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Analyze radar processing results')
    parser.add_argument('--output', required=True, help='Output directory containing processed data')
    parser.add_argument('--frame', type=int, default=0, help='Frame index to analyze')
    parser.add_argument('--antenna', type=int, default=0, help='Antenna index to analyze')
    parser.add_argument('--compare', action='store_true', help='Compare multiple frames')
    parser.add_argument('--insights', action='store_true', help='Generate comprehensive insights report')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = RadarDataAnalyzer(args.output)
    
    if args.insights:
        analyzer.generate_insights_report()
    elif args.compare:
        analyzer.compare_frames()
    else:
        # Analyze specific frame
        analyzer.analyze_peaks(args.frame)
        analyzer.analyze_rds_quality(args.frame, args.antenna)
    
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
