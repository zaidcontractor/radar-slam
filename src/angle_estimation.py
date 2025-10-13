"""
Angle of Arrival (AoA) Estimation Module

This module implements the angle extraction methods described in the paper
"3-D Ego-Motion Estimation Using Multi-Channel FMCW Radar" (Yuan et al. 2023).

The module provides:
1. Spatial signature extraction from RDS data
2. MUSIC algorithm for AoA estimation
3. ESPRIT algorithm as alternative
4. Target-specific angle estimation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, svd
from scipy.optimize import minimize_scalar
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class AngleEstimator:
    """
    Angle of Arrival estimation for multi-channel FMCW radar.
    
    Implements spatial signature extraction and AoA estimation methods
    as described in the paper.
    """
    
    def __init__(self,
                 fc: float = 77e9,
                 antenna_spacing: float = None,
                 num_antennas: int = 8,
                 search_range: Tuple[float, float] = (-90, 90),
                 search_resolution: float = 0.5):
        """
        Initialize angle estimator.
        
        Args:
            fc: Carrier frequency (Hz)
            antenna_spacing: Spacing between antenna elements (m)
            num_antennas: Number of antenna elements
            search_range: Azimuth search range (degrees)
            search_resolution: Search resolution (degrees)
        """
        self.fc = fc
        self.c = 3e8  # Speed of light
        self.lambda_c = self.c / self.fc
        self.antenna_spacing = antenna_spacing or (self.lambda_c / 2)
        self.num_antennas = num_antennas
        self.search_range = search_range
        self.search_resolution = search_resolution
        
        # Antenna array geometry (ULA)
        self.antenna_positions = np.arange(self.num_antennas) * self.antenna_spacing
        
        # Search grid
        self.azimuth_grid = np.arange(search_range[0], search_range[1] + search_resolution, 
                                    search_resolution)
        
        logger.info(f"Initialized angle estimator:")
        logger.info(f"  Antennas: {self.num_antennas}, spacing: {self.antenna_spacing*1000:.1f} mm")
        logger.info(f"  Search range: {search_range[0]}° to {search_range[1]}°")
        logger.info(f"  Search resolution: {search_resolution}°")
    
    def extract_spatial_signature(self, 
                                 rds: np.ndarray,
                                 range_bin: int,
                                 doppler_bin: int) -> np.ndarray:
        """
        Extract spatial signature for a specific target.
        
        Args:
            rds: RDS matrix [num_antennas, range_bins, doppler_bins]
            range_bin: Range bin index
            doppler_bin: Doppler bin index
            
        Returns:
            Spatial signature vector [num_antennas]
        """
        # Extract complex values across antennas for this range-Doppler bin
        spatial_signature = rds[:, range_bin, doppler_bin]
        
        # Normalize to unit power
        power = np.sum(np.abs(spatial_signature)**2)
        if power > 0:
            spatial_signature = spatial_signature / np.sqrt(power)
        
        return spatial_signature
    
    def generate_steering_vector(self, azimuth_deg: float) -> np.ndarray:
        """
        Generate steering vector for given azimuth angle.
        
        Args:
            azimuth_deg: Azimuth angle in degrees
            
        Returns:
            Steering vector [num_antennas]
        """
        azimuth_rad = np.radians(azimuth_deg)
        
        # Phase shifts due to antenna positions
        phases = 2 * np.pi * self.antenna_positions * np.sin(azimuth_rad) / self.lambda_c
        
        return np.exp(1j * phases)
    
    def music_spectrum(self, 
                      spatial_signature: np.ndarray,
                      num_sources: int = 1) -> np.ndarray:
        """
        Compute MUSIC spectrum for AoA estimation.
        
        Args:
            spatial_signature: Spatial signature vector [num_antennas]
            num_sources: Number of signal sources
            
        Returns:
            MUSIC spectrum [num_angles]
        """
        # For single snapshot, use the signature directly
        # In practice, you might want to collect multiple snapshots
        # and compute covariance matrix
        
        # Create a simple covariance matrix from the signature
        R = np.outer(spatial_signature, spatial_signature.conj())
        
        # Eigen decomposition
        eigenvals, eigenvecs = eigh(R)
        
        # Sort eigenvalues in descending order
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Noise subspace (smallest eigenvalues)
        noise_subspace = eigenvecs[:, num_sources:]
        
        # Compute MUSIC spectrum
        music_spectrum = np.zeros(len(self.azimuth_grid))
        
        for i, azimuth in enumerate(self.azimuth_grid):
            steering_vector = self.generate_steering_vector(azimuth)
            
            # MUSIC pseudo-spectrum
            denominator = np.abs(steering_vector.conj().T @ noise_subspace @ noise_subspace.conj().T @ steering_vector)
            
            if denominator > 1e-12:
                music_spectrum[i] = 1.0 / denominator
            else:
                music_spectrum[i] = 0.0
        
        return music_spectrum
    
    def estimate_angle_music(self, 
                           spatial_signature: np.ndarray,
                           num_sources: int = 1) -> Tuple[float, np.ndarray]:
        """
        Estimate angle using MUSIC algorithm.
        
        Args:
            spatial_signature: Spatial signature vector [num_antennas]
            num_sources: Number of signal sources
            
        Returns:
            Tuple of (estimated_angle, music_spectrum)
        """
        # Compute MUSIC spectrum
        music_spectrum = self.music_spectrum(spatial_signature, num_sources)
        
        # Find peak
        peak_idx = np.argmax(music_spectrum)
        estimated_angle = self.azimuth_grid[peak_idx]
        
        return estimated_angle, music_spectrum
    
    def estimate_angle_esprit(self, 
                            spatial_signature: np.ndarray,
                            num_sources: int = 1) -> float:
        """
        Estimate angle using ESPRIT algorithm.
        
        Args:
            spatial_signature: Spatial signature vector [num_antennas]
            num_sources: Number of signal sources
            
        Returns:
            Estimated angle in degrees
        """
        # For ESPRIT, we need multiple snapshots or a covariance matrix
        # Here we use a simplified approach with the spatial signature
        
        # Create two subarrays (first M-1 and last M-1 antennas)
        subarray1 = spatial_signature[:-1]
        subarray2 = spatial_signature[1:]
        
        # Solve for rotation matrix
        try:
            # SVD of concatenated subarrays
            U, s, Vh = svd(np.column_stack([subarray1, subarray2]))
            
            # Extract signal subspace
            signal_subspace = U[:, :num_sources]
            
            # Split into two parts
            U1 = signal_subspace[:-1, :]
            U2 = signal_subspace[1:, :]
            
            # Solve for rotation matrix
            Phi = np.linalg.pinv(U1) @ U2
            
            # Extract eigenvalues (phase shifts)
            eigenvals = np.linalg.eigvals(Phi)
            
            # Convert to angle
            phase_shift = np.angle(eigenvals[0])
            angle_rad = np.arcsin(phase_shift * self.lambda_c / (2 * np.pi * self.antenna_spacing))
            angle_deg = np.degrees(angle_rad)
            
            return angle_deg
            
        except Exception as e:
            logger.warning(f"ESPRIT failed: {e}")
            return 0.0
    
    def estimate_angle_beamforming(self, 
                                  spatial_signature: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Estimate angle using beamforming (conventional method).
        
        Args:
            spatial_signature: Spatial signature vector [num_antennas]
            
        Returns:
            Tuple of (estimated_angle, beamforming_spectrum)
        """
        # Compute beamforming spectrum
        beamforming_spectrum = np.zeros(len(self.azimuth_grid))
        
        for i, azimuth in enumerate(self.azimuth_grid):
            steering_vector = self.generate_steering_vector(azimuth)
            
            # Beamforming output power
            beamforming_spectrum[i] = np.abs(steering_vector.conj().T @ spatial_signature)**2
        
        # Find peak
        peak_idx = np.argmax(beamforming_spectrum)
        estimated_angle = self.azimuth_grid[peak_idx]
        
        return estimated_angle, beamforming_spectrum
    
    def process_targets(self, 
                       rds: np.ndarray,
                       peak_info: Dict,
                       method: str = 'music') -> List[Dict]:
        """
        Process all detected targets to estimate their angles.
        
        Args:
            rds: RDS matrix [num_antennas, range_bins, doppler_bins]
            peak_info: Peak information from RDS processing
            method: Estimation method ('music', 'esprit', 'beamforming')
            
        Returns:
            List of target angle estimates
        """
        targets = []
        
        for peak in peak_info['peaks']:
            try:
                # Extract spatial signature
                spatial_signature = self.extract_spatial_signature(
                    rds, peak['range_bin'], peak['doppler_bin']
                )
                
                # Estimate angle based on method
                if method == 'music':
                    angle, spectrum = self.estimate_angle_music(spatial_signature)
                elif method == 'esprit':
                    angle = self.estimate_angle_esprit(spatial_signature)
                    spectrum = None
                elif method == 'beamforming':
                    angle, spectrum = self.estimate_angle_beamforming(spatial_signature)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                # Create target entry
                target = {
                    'range_m': peak['range_m'],
                    'doppler_hz': peak['doppler_hz'],
                    'power_db': peak['power_db'],
                    'azimuth_deg': angle,
                    'azimuth_rad': np.radians(angle),
                    'antenna': peak['antenna'],
                    'range_bin': peak['range_bin'],
                    'doppler_bin': peak['doppler_bin'],
                    'spatial_signature': spatial_signature,
                    'spectrum': spectrum
                }
                
                targets.append(target)
                
            except Exception as e:
                logger.warning(f"Error processing target: {e}")
                continue
        
        logger.info(f"Processed {len(targets)} targets using {method}")
        return targets
    
    def visualize_angle_spectrum(self, 
                               targets: List[Dict],
                               save_path: Optional[str] = None) -> None:
        """
        Visualize angle estimation results.
        
        Args:
            targets: List of target angle estimates
            save_path: Path to save plot
        """
        if not targets:
            logger.warning("No targets to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Angle distribution
        angles = [t['azimuth_deg'] for t in targets]
        axes[0, 0].hist(angles, bins=20, alpha=0.7)
        axes[0, 0].set_xlabel('Azimuth Angle (degrees)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Angle Distribution')
        axes[0, 0].grid(True)
        
        # Plot 2: Range vs Angle
        ranges = [t['range_m'] for t in targets]
        axes[0, 1].scatter(angles, ranges, c=[t['power_db'] for t in targets], 
                          cmap='viridis', alpha=0.7)
        axes[0, 1].set_xlabel('Azimuth Angle (degrees)')
        axes[0, 1].set_ylabel('Range (m)')
        axes[0, 1].set_title('Range vs Angle')
        axes[0, 1].grid(True)
        
        # Plot 3: Power vs Angle
        powers = [t['power_db'] for t in targets]
        axes[1, 0].scatter(angles, powers, alpha=0.7)
        axes[1, 0].set_xlabel('Azimuth Angle (degrees)')
        axes[1, 0].set_ylabel('Power (dB)')
        axes[1, 0].set_title('Power vs Angle')
        axes[1, 0].grid(True)
        
        # Plot 4: MUSIC spectrum (if available)
        if targets[0]['spectrum'] is not None:
            spectrum = targets[0]['spectrum']
            axes[1, 1].plot(self.azimuth_grid, 10 * np.log10(spectrum + 1e-12))
            axes[1, 1].set_xlabel('Azimuth Angle (degrees)')
            axes[1, 1].set_ylabel('MUSIC Spectrum (dB)')
            axes[1, 1].set_title('MUSIC Spectrum')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def extract_angles_from_rds(rds_path: str,
                           peak_info_path: str,
                           output_path: str,
                           method: str = 'music',
                           radar_params: Dict = None) -> Dict:
    """
    Extract angles from RDS data.
    
    Args:
        rds_path: Path to RDS file
        peak_info_path: Path to peak information file
        output_path: Path to save angle estimates
        method: Angle estimation method
        radar_params: Radar parameters
        
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
    
    # Initialize angle estimator
    estimator = AngleEstimator(**radar_params)
    
    # Process targets
    targets = estimator.process_targets(rds, peak_info, method)
    
    # Save results
    np.savez(output_path, targets=targets, radar_params=radar_params)
    
    logger.info(f"Saved angle estimates for {len(targets)} targets")
    
    return {
        'num_targets': len(targets),
        'method': method,
        'targets': targets
    }


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract angles from RDS data')
    parser.add_argument('--rds', required=True, help='Path to RDS file')
    parser.add_argument('--peaks', required=True, help='Path to peak info file')
    parser.add_argument('--out', required=True, help='Output path for angles')
    parser.add_argument('--method', choices=['music', 'esprit', 'beamforming'], 
                       default='music', help='Angle estimation method')
    
    args = parser.parse_args()
    
    # Process angles
    results = extract_angles_from_rds(args.rds, args.peaks, args.out, args.method)
    print(f"Angle extraction complete: {results}")
