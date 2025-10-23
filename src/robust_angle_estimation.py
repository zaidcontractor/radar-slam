"""
Robust Angle Estimation Module

This module implements enhanced angle estimation with:
1. Temporal smoothing for angle estimates
2. Confidence metrics for angle quality
3. Multi-path and interference handling
4. Real-time processing optimizations
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, svd
from scipy.signal import savgol_filter
from scipy.optimize import minimize_scalar
from typing import Dict, List, Tuple, Optional
import logging
from collections import deque
import time

logger = logging.getLogger(__name__)

class RobustAngleEstimator:
    """
    Enhanced angle estimator with temporal smoothing and confidence metrics.
    
    Features:
    - Temporal smoothing for stable angle estimates
    - Confidence metrics for angle quality assessment
    - Multi-path interference handling
    - Real-time processing optimizations
    """
    
    def __init__(self,
                 fc: float = 77e9,
                 antenna_spacing: float = None,
                 num_antennas: int = 8,
                 search_range: Tuple[float, float] = (-90, 90),
                 search_resolution: float = 1.0,
                 temporal_window: int = 5,
                 confidence_threshold: float = 0.7,
                 smoothing_factor: float = 0.3,
                 max_targets: int = 100):
        """
        Initialize robust angle estimator.
        
        Args:
            fc: Carrier frequency (Hz)
            antenna_spacing: Spacing between antenna elements (m)
            num_antennas: Number of antenna elements
            search_range: Azimuth search range (degrees)
            search_resolution: Search resolution (degrees)
            temporal_window: Window size for temporal smoothing
            confidence_threshold: Minimum confidence for valid estimates
            smoothing_factor: Smoothing strength (0-1)
            max_targets: Maximum number of targets to process
        """
        self.fc = fc
        self.c = 3e8  # Speed of light
        self.lambda_c = self.c / self.fc
        self.antenna_spacing = antenna_spacing or (self.lambda_c / 2)
        self.num_antennas = num_antennas
        self.search_range = search_range
        self.search_resolution = search_resolution
        self.temporal_window = temporal_window
        self.confidence_threshold = confidence_threshold
        self.smoothing_factor = smoothing_factor
        self.max_targets = max_targets
        
        # Antenna array geometry (ULA)
        self.antenna_positions = np.arange(self.num_antennas) * self.antenna_spacing
        
        # Search grid
        self.azimuth_grid = np.arange(search_range[0], search_range[1] + search_resolution, 
                                    search_resolution)
        
        # Temporal buffers for smoothing
        self.angle_history = {}  # Target ID -> deque of angles
        self.confidence_history = {}  # Target ID -> deque of confidences
        self.target_counter = 0
        
        logger.info(f"Initialized robust angle estimator:")
        logger.info(f"  Temporal window: {temporal_window}")
        logger.info(f"  Confidence threshold: {confidence_threshold}")
        logger.info(f"  Smoothing factor: {smoothing_factor}")
        logger.info(f"  Max targets: {max_targets}")
    
    def compute_angle_confidence(self, 
                               spatial_signature: np.ndarray,
                               estimated_angle: float) -> float:
        """
        Compute confidence metric for angle estimate.
        
        Args:
            spatial_signature: Spatial signature vector
            estimated_angle: Estimated angle in degrees
            
        Returns:
            Confidence score (0-1)
        """
        # Generate steering vector for estimated angle
        steering_vector = self.generate_steering_vector(estimated_angle)
        
        # Compute correlation with spatial signature
        correlation = np.abs(steering_vector.conj().T @ spatial_signature)
        
        # Normalize by signal power
        signal_power = np.sum(np.abs(spatial_signature)**2)
        if signal_power > 0:
            normalized_correlation = correlation / np.sqrt(signal_power)
        else:
            normalized_correlation = 0.0
        
        # Compute angular spread (how sharp the peak is)
        angle_rad = np.radians(estimated_angle)
        phases = 2 * np.pi * self.antenna_positions * np.sin(angle_rad) / self.lambda_c
        expected_phases = np.angle(steering_vector)
        actual_phases = np.angle(spatial_signature)
        
        # Phase consistency (lower is better)
        phase_error = np.mean(np.abs(np.angle(np.exp(1j * (actual_phases - expected_phases)))))
        phase_consistency = np.exp(-phase_error)  # Convert to 0-1 scale
        
        # Signal-to-noise ratio estimate
        signal_power = np.abs(spatial_signature)**2
        noise_floor = np.percentile(signal_power, 20)  # Estimate noise floor
        if noise_floor > 0:
            snr_estimate = np.mean(signal_power) / noise_floor
            snr_confidence = min(1.0, np.log10(snr_estimate) / 3.0)  # Normalize to 0-1
        else:
            snr_confidence = 0.0
        
        # Combine metrics
        confidence = (normalized_correlation * 0.4 + 
                    phase_consistency * 0.3 + 
                    snr_confidence * 0.3)
        
        return min(1.0, max(0.0, confidence))
    
    def detect_multipath_interference(self, 
                                    spatial_signature: np.ndarray) -> Dict:
        """
        Detect multi-path interference and estimate number of sources.
        
        Args:
            spatial_signature: Spatial signature vector
            
        Returns:
            Dictionary with interference analysis
        """
        # Compute covariance matrix
        R = np.outer(spatial_signature, spatial_signature.conj())
        
        # Eigen decomposition
        eigenvals, eigenvecs = eigh(R)
        eigenvals = np.real(eigenvals)  # Ensure real eigenvalues
        
        # Sort eigenvalues in descending order
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        
        # Estimate number of sources using MDL (Minimum Description Length)
        N = len(eigenvals)
        mdl_values = []
        
        for k in range(1, min(N, 5)):  # Test up to 4 sources
            # MDL criterion
            L = N - k
            if L > 0:
                # Signal eigenvalues
                signal_eigenvals = eigenvals[:k]
                # Noise eigenvalues
                noise_eigenvals = eigenvals[k:]
                
                if len(noise_eigenvals) > 0 and np.mean(noise_eigenvals) > 0:
                    # Geometric mean of noise eigenvalues
                    noise_mean = np.mean(noise_eigenvals)
                    # Arithmetic mean of noise eigenvalues
                    noise_arithmetic = np.mean(noise_eigenvals)
                    
                    # MDL calculation
                    mdl = -L * np.log(noise_mean / noise_arithmetic) + 0.5 * k * (2*N - k) * np.log(L)
                    mdl_values.append(mdl)
                else:
                    mdl_values.append(float('inf'))
            else:
                mdl_values.append(float('inf'))
        
        # Find minimum MDL
        if mdl_values and min(mdl_values) != float('inf'):
            num_sources = np.argmin(mdl_values) + 1
        else:
            num_sources = 1
        
        # Compute interference metrics
        if len(eigenvals) > 1:
            # Eigenvalue ratio (signal to noise)
            signal_power = np.sum(eigenvals[:num_sources])
            noise_power = np.sum(eigenvals[num_sources:])
            if noise_power > 0:
                snr_ratio = signal_power / noise_power
            else:
                snr_ratio = float('inf')
            
            # Condition number
            condition_number = eigenvals[0] / eigenvals[-1] if eigenvals[-1] > 0 else float('inf')
        else:
            snr_ratio = float('inf')
            condition_number = float('inf')
        
        return {
            'num_sources': num_sources,
            'snr_ratio': snr_ratio,
            'condition_number': condition_number,
            'eigenvalues': eigenvals,
            'is_multipath': num_sources > 1,
            'interference_level': min(1.0, 1.0 / snr_ratio) if snr_ratio > 0 else 1.0
        }
    
    def estimate_angle_robust(self, 
                            spatial_signature: np.ndarray,
                            target_id: str = None) -> Dict:
        """
        Robust angle estimation with confidence metrics and interference detection.
        
        Args:
            spatial_signature: Spatial signature vector
            target_id: Target identifier for temporal smoothing
            
        Returns:
            Dictionary with angle estimate and quality metrics
        """
        # Detect interference
        interference_analysis = self.detect_multipath_interference(spatial_signature)
        
        # Fast beamforming for initial estimate
        beamforming_spectrum = np.zeros(len(self.azimuth_grid))
        
        for i, azimuth in enumerate(self.azimuth_grid):
            steering_vector = self.generate_steering_vector(azimuth)
            beamforming_spectrum[i] = np.abs(steering_vector.conj().T @ spatial_signature)**2
        
        # Find peak
        peak_idx = np.argmax(beamforming_spectrum)
        initial_angle = self.azimuth_grid[peak_idx]
        
        # Compute confidence
        confidence = self.compute_angle_confidence(spatial_signature, initial_angle)
        
        # Apply temporal smoothing if target_id provided
        if target_id is not None:
            smoothed_angle, smoothed_confidence = self.apply_temporal_smoothing(
                target_id, initial_angle, confidence
            )
        else:
            smoothed_angle = initial_angle
            smoothed_confidence = confidence
        
        # Determine if estimate is reliable
        is_reliable = (smoothed_confidence >= self.confidence_threshold and 
                      not interference_analysis['is_multipath'])
        
        return {
            'angle_deg': smoothed_angle,
            'angle_rad': np.radians(smoothed_angle),
            'confidence': smoothed_confidence,
            'is_reliable': is_reliable,
            'interference_analysis': interference_analysis,
            'spectrum': beamforming_spectrum,
            'initial_angle': initial_angle,
            'smoothing_applied': target_id is not None
        }
    
    def apply_temporal_smoothing(self, 
                               target_id: str,
                               new_angle: float,
                               new_confidence: float) -> Tuple[float, float]:
        """
        Apply temporal smoothing to angle estimates.
        
        Args:
            target_id: Target identifier
            new_angle: New angle estimate
            new_confidence: New confidence score
            
        Returns:
            Tuple of (smoothed_angle, smoothed_confidence)
        """
        # Initialize history for new targets
        if target_id not in self.angle_history:
            self.angle_history[target_id] = deque(maxlen=self.temporal_window)
            self.confidence_history[target_id] = deque(maxlen=self.temporal_window)
        
        # Add new measurements
        self.angle_history[target_id].append(new_angle)
        self.confidence_history[target_id].append(new_confidence)
        
        # Apply smoothing
        if len(self.angle_history[target_id]) >= 2:
            # Weighted average based on confidence
            angles = np.array(self.angle_history[target_id])
            confidences = np.array(self.confidence_history[target_id])
            
            # Normalize weights
            weights = confidences / np.sum(confidences) if np.sum(confidences) > 0 else np.ones_like(confidences) / len(confidences)
            
            # Handle angle wrapping
            angles_rad = np.radians(angles)
            cos_angles = np.cos(angles_rad)
            sin_angles = np.sin(angles_rad)
            
            # Weighted circular mean
            mean_cos = np.sum(weights * cos_angles)
            mean_sin = np.sum(weights * sin_angles)
            smoothed_angle_rad = np.arctan2(mean_sin, mean_cos)
            smoothed_angle = np.degrees(smoothed_angle_rad)
            
            # Apply additional smoothing factor
            if len(self.angle_history[target_id]) > 1:
                prev_angle = self.angle_history[target_id][-2]
                smoothed_angle = (self.smoothing_factor * smoothed_angle + 
                                (1 - self.smoothing_factor) * prev_angle)
            
            # Smooth confidence
            smoothed_confidence = np.mean(confidences)
        else:
            smoothed_angle = new_angle
            smoothed_confidence = new_confidence
        
        return smoothed_angle, smoothed_confidence
    
    def generate_steering_vector(self, azimuth_deg: float) -> np.ndarray:
        """
        Generate steering vector for given azimuth angle.
        
        Args:
            azimuth_deg: Azimuth angle in degrees
            
        Returns:
            Steering vector [num_antennas]
        """
        azimuth_rad = np.radians(azimuth_deg)
        phases = 2 * np.pi * self.antenna_positions * np.sin(azimuth_rad) / self.lambda_c
        return np.exp(1j * phases)
    
    def process_targets_robust(self, 
                             rds: np.ndarray,
                             peak_info: Dict,
                             frame_timestamp: float = None) -> List[Dict]:
        """
        Process targets with robust angle estimation.
        
        Args:
            rds: RDS matrix [num_antennas, range_bins, doppler_bins]
            peak_info: Peak information from RDS processing
            frame_timestamp: Frame timestamp for temporal processing
            
        Returns:
            List of robust target angle estimates
        """
        # Filter targets by power
        peaks = peak_info['peaks']
        filtered_peaks = [p for p in peaks if p['power_db'] > -25.0]  # Power threshold
        filtered_peaks.sort(key=lambda x: x['power_db'], reverse=True)
        filtered_peaks = filtered_peaks[:self.max_targets]
        
        targets = []
        
        for i, peak in enumerate(filtered_peaks):
            try:
                # Extract spatial signature
                spatial_signature = rds[:, peak['range_bin'], peak['doppler_bin']]
                
                # Normalize
                power = np.sum(np.abs(spatial_signature)**2)
                if power > 0:
                    spatial_signature = spatial_signature / np.sqrt(power)
                
                # Generate target ID for temporal tracking
                target_id = f"target_{peak['range_bin']}_{peak['doppler_bin']}"
                
                # Robust angle estimation
                angle_result = self.estimate_angle_robust(spatial_signature, target_id)
                
                # Only include reliable estimates
                if angle_result['is_reliable']:
                    target = {
                        'range_m': peak['range_m'],
                        'doppler_hz': peak['doppler_hz'],
                        'power_db': peak['power_db'],
                        'azimuth_deg': angle_result['angle_deg'],
                        'azimuth_rad': angle_result['angle_rad'],
                        'confidence': angle_result['confidence'],
                        'is_reliable': angle_result['is_reliable'],
                        'interference_analysis': angle_result['interference_analysis'],
                        'antenna': peak['antenna'],
                        'range_bin': peak['range_bin'],
                        'doppler_bin': peak['doppler_bin'],
                        'spatial_signature': spatial_signature,
                        'target_id': target_id,
                        'timestamp': frame_timestamp or time.time()
                    }
                    
                    targets.append(target)
                
            except Exception as e:
                logger.warning(f"Error processing target: {e}")
                continue
        
        logger.info(f"Processed {len(targets)} reliable targets (filtered from {len(filtered_peaks)})")
        return targets
    
    def get_target_statistics(self) -> Dict:
        """
        Get statistics about target processing.
        
        Returns:
            Dictionary with processing statistics
        """
        total_targets = len(self.angle_history)
        active_targets = sum(1 for history in self.angle_history.values() if len(history) > 0)
        
        # Compute average confidence
        all_confidences = []
        for confidences in self.confidence_history.values():
            all_confidences.extend(confidences)
        
        avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
        
        return {
            'total_targets_tracked': total_targets,
            'active_targets': active_targets,
            'average_confidence': avg_confidence,
            'temporal_window_size': self.temporal_window,
            'confidence_threshold': self.confidence_threshold
        }
    
    def visualize_angle_quality(self, 
                              targets: List[Dict],
                              save_path: Optional[str] = None) -> None:
        """
        Visualize angle estimation quality metrics.
        
        Args:
            targets: List of target angle estimates
            save_path: Path to save plot
        """
        if not targets:
            logger.warning("No targets to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Angle distribution
        angles = [t['azimuth_deg'] for t in targets]
        confidences = [t['confidence'] for t in targets]
        
        axes[0, 0].hist(angles, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Azimuth Angle (degrees)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Angle Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Confidence distribution
        axes[0, 1].hist(confidences, bins=20, alpha=0.7, edgecolor='black', color='green')
        axes[0, 1].set_xlabel('Confidence Score')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Confidence Distribution')
        axes[0, 1].axvline(self.confidence_threshold, color='red', linestyle='--', 
                          label=f'Threshold: {self.confidence_threshold}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Angle vs Confidence
        axes[1, 0].scatter(angles, confidences, alpha=0.7, s=50)
        axes[1, 0].set_xlabel('Azimuth Angle (degrees)')
        axes[1, 0].set_ylabel('Confidence Score')
        axes[1, 0].set_title('Angle vs Confidence')
        axes[1, 0].axhline(self.confidence_threshold, color='red', linestyle='--')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Interference analysis
        interference_levels = [t['interference_analysis']['interference_level'] for t in targets]
        multipath_count = sum(1 for t in targets if t['interference_analysis']['is_multipath'])
        
        axes[1, 1].hist(interference_levels, bins=20, alpha=0.7, edgecolor='black', color='orange')
        axes[1, 1].set_xlabel('Interference Level')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title(f'Interference Analysis (Multipath: {multipath_count})')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"Total Targets: {len(targets)}\n"
        stats_text += f"Avg Confidence: {np.mean(confidences):.3f}\n"
        stats_text += f"Multipath Targets: {multipath_count}\n"
        stats_text += f"Reliable Targets: {sum(1 for t in targets if t['is_reliable'])}"
        
        fig.text(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def extract_angles_robust(rds_path: str,
                         peak_info_path: str,
                         output_path: str,
                         radar_params: Dict = None,
                         temporal_window: int = 5,
                         confidence_threshold: float = 0.7) -> Dict:
    """
    Extract angles using robust estimation with temporal smoothing.
    
    Args:
        rds_path: Path to RDS file
        peak_info_path: Path to peak info file
        output_path: Path to save angle estimates
        radar_params: Radar parameters
        temporal_window: Window size for temporal smoothing
        confidence_threshold: Minimum confidence threshold
        
    Returns:
        Processing results
    """
    # Load RDS data
    rds = np.load(rds_path)
    peak_data = np.load(peak_info_path, allow_pickle=True)
    peak_info = dict(peak_data)
    
    logger.info(f"Loaded RDS: {rds.shape}")
    logger.info(f"Found {len(peak_info['peaks'])} peaks")
    
    # Default radar parameters
    if radar_params is None:
        radar_params = {
            'fc': 77e9,
            'antenna_spacing': 3e8 / (2 * 77e9),
            'num_antennas': 8
        }
    
    # Initialize robust angle estimator
    estimator = RobustAngleEstimator(
        **radar_params,
        temporal_window=temporal_window,
        confidence_threshold=confidence_threshold
    )
    
    # Process targets
    targets = estimator.process_targets_robust(rds, peak_info)
    
    # Get statistics
    stats = estimator.get_target_statistics()
    
    # Save results
    np.savez(output_path, 
             targets=targets, 
             radar_params=radar_params,
             statistics=stats)
    
    logger.info(f"Saved robust angle estimates for {len(targets)} targets")
    logger.info(f"Statistics: {stats}")
    
    return {
        'num_targets': len(targets),
        'statistics': stats,
        'targets': targets
    }


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract angles with robust estimation')
    parser.add_argument('--rds', required=True, help='Path to RDS file')
    parser.add_argument('--peaks', required=True, help='Path to peak info file')
    parser.add_argument('--out', required=True, help='Output path for angles')
    parser.add_argument('--temporal-window', type=int, default=5, help='Temporal window size')
    parser.add_argument('--confidence-threshold', type=float, default=0.7, help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Process angles
    results = extract_angles_robust(args.rds, args.peaks, args.out, 
                                   temporal_window=args.temporal_window,
                                   confidence_threshold=args.confidence_threshold)
    print(f"Robust angle extraction complete: {results}")
