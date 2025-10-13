#!/usr/bin/env python3
"""
Raw FMCW Signal Synthesis from RadarScenes Point Clouds

This script synthesizes raw FMCW baseband signals from RadarScenes point cloud data,
following the signal model described in the paper "3-D Ego-Motion Estimation Using 
Multi-Channel FMCW Radar" (Yuan et al. 2023).

The synthesis process converts scatterers (point clouds) to complex time-domain 
baseband signals as would be received by a multi-channel FMCW radar system.
"""

import numpy as np
import pandas as pd
import h5py
import json
import os
import argparse
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FMCWRadarSimulator:
    """
    Simulates raw FMCW radar signals from point cloud scatterers.
    
    This class implements the signal model from the paper, converting
    RadarScenes point clouds into synthetic raw FMCW baseband signals.
    """
    
    def __init__(self, 
                 fc: float = 77e9,  # Carrier frequency (Hz)
                 bandwidth: float = 1e9,  # Bandwidth (Hz)
                 chirp_duration: float = 40e-6,  # Chirp duration (s)
                 pri: float = 100e-6,  # Pulse repetition interval (s)
                 num_chirps: int = 64,  # Number of chirps per frame
                 num_antennas: int = 8,  # Number of antenna elements
                 antenna_spacing: float = None,  # Antenna spacing (m)
                 sampling_rate: float = 10e6,  # Sampling rate (Hz)
                 noise_power: float = 0.01):  # Noise power
        """
        Initialize FMCW radar simulator with radar parameters.
        
        Args:
            fc: Carrier frequency (Hz)
            bandwidth: Chirp bandwidth (Hz)
            chirp_duration: Duration of each chirp (s)
            pri: Pulse repetition interval (s)
            num_chirps: Number of chirps per frame
            num_antennas: Number of antenna elements
            antenna_spacing: Spacing between antenna elements (m)
            sampling_rate: ADC sampling rate (Hz)
            noise_power: Additive noise power
        """
        # Radar parameters
        self.fc = fc
        self.bandwidth = bandwidth
        self.chirp_duration = chirp_duration
        self.pri = pri
        self.num_chirps = num_chirps
        self.num_antennas = num_antennas
        self.sampling_rate = sampling_rate
        self.noise_power = noise_power
        
        # Derived parameters
        self.c = 3e8  # Speed of light
        self.lambda_c = self.c / self.fc  # Wavelength
        self.antenna_spacing = antenna_spacing or (self.lambda_c / 2)  # Half-wavelength spacing
        
        # Sampling parameters
        self.samples_per_chirp = int(self.chirp_duration * self.sampling_rate)
        self.chirp_rate = self.bandwidth / self.chirp_duration  # Hz/s
        
        # Antenna array geometry (ULA)
        self.antenna_positions = np.arange(self.num_antennas) * self.antenna_spacing
        
        logger.info(f"Initialized FMCW radar simulator:")
        logger.info(f"  Carrier frequency: {self.fc/1e9:.1f} GHz")
        logger.info(f"  Bandwidth: {self.bandwidth/1e9:.1f} GHz")
        logger.info(f"  Chirp duration: {self.chirp_duration*1e6:.1f} μs")
        logger.info(f"  PRI: {self.pri*1e6:.1f} μs")
        logger.info(f"  Antennas: {self.num_antennas}, spacing: {self.antenna_spacing*1000:.1f} mm")
    
    def generate_chirp_signal(self, t: np.ndarray) -> np.ndarray:
        """
        Generate the transmitted chirp signal.
        
        Args:
            t: Time vector (s)
            
        Returns:
            Complex chirp signal
        """
        # Linear frequency modulation
        phase = 2 * np.pi * (self.fc * t + 0.5 * self.chirp_rate * t**2)
        return np.exp(1j * phase)
    
    def compute_target_response(self, 
                              range_m: float, 
                              azimuth_rad: float, 
                              elevation_rad: float,
                              rcs_db: float,
                              velocity_radial: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the radar response from a single target.
        
        Args:
            range_m: Target range (m)
            azimuth_rad: Target azimuth angle (rad)
            elevation_rad: Target elevation angle (rad)
            rcs_db: Radar cross section (dBsm)
            velocity_radial: Radial velocity (m/s)
            
        Returns:
            Tuple of (time_delay, complex_amplitude)
        """
        # Time delay due to range
        time_delay = 2 * range_m / self.c
        
        # RCS to amplitude conversion
        rcs_linear = 10**(rcs_db / 10)  # Convert dBsm to linear
        amplitude = np.sqrt(rcs_linear) / (4 * np.pi * range_m**2)
        
        # Doppler phase shift
        doppler_phase = 4 * np.pi * velocity_radial * self.fc / self.c
        
        # Direction vector
        direction_vector = np.array([
            np.cos(elevation_rad) * np.cos(azimuth_rad),
            np.cos(elevation_rad) * np.sin(azimuth_rad),
            np.sin(elevation_rad)
        ])
        
        # Antenna phase shifts
        antenna_phases = np.zeros(self.num_antennas, dtype=complex)
        for i in range(self.num_antennas):
            # Phase shift due to antenna position
            antenna_phase = 2 * np.pi * self.antenna_positions[i] * np.sin(azimuth_rad) / self.lambda_c
            antenna_phases[i] = amplitude * np.exp(1j * (doppler_phase + antenna_phase))
        
        return time_delay, antenna_phases
    
    def synthesize_frame(self, 
                        scatterers: pd.DataFrame,
                        frame_idx: int = 0) -> np.ndarray:
        """
        Synthesize raw FMCW signals for a single frame.
        
        Args:
            scatterers: DataFrame with columns [range, azimuth, elevation, rcs, velocity]
            frame_idx: Frame index for temporal variation
            
        Returns:
            Complex baseband signals [num_antennas, num_chirps, samples_per_chirp]
        """
        # Initialize output array
        signals = np.zeros((self.num_antennas, self.num_chirps, self.samples_per_chirp), 
                          dtype=complex)
        
        # Time vector for each chirp
        t_chirp = np.linspace(0, self.chirp_duration, self.samples_per_chirp)
        
        # Generate reference chirp
        ref_chirp = self.generate_chirp_signal(t_chirp)
        
        # Process each scatterer
        for _, scatterer in scatterers.iterrows():
            try:
                # Extract scatterer parameters
                range_m = scatterer.get('range_sc', 0.0)
                azimuth_rad = scatterer.get('azimuth_sc', 0.0)
                elevation_rad = 0.0  # Assume ground level for RadarScenes
                rcs_db = scatterer.get('rcs', -10.0)
                velocity_radial = scatterer.get('vr', 0.0)
                
                # Skip invalid scatterers
                if range_m <= 0 or not np.isfinite([range_m, azimuth_rad, rcs_db, velocity_radial]).all():
                    continue
                
                # Compute target response
                time_delay, antenna_phases = self.compute_target_response(
                    range_m, azimuth_rad, elevation_rad, rcs_db, velocity_radial
                )
                
                # Generate received signal for each chirp and antenna
                for chirp_idx in range(self.num_chirps):
                    # Time offset for this chirp
                    chirp_start_time = chirp_idx * self.pri
                    
                    for ant_idx in range(self.num_antennas):
                        # Delayed chirp signal
                        t_delayed = t_chirp - time_delay
                        valid_idx = (t_delayed >= 0) & (t_delayed <= self.chirp_duration)
                        
                        if np.any(valid_idx):
                            # Generate delayed chirp
                            delayed_chirp = self.generate_chirp_signal(t_delayed[valid_idx])
                            
                            # Mix with reference (dechirp)
                            baseband = delayed_chirp * np.conj(ref_chirp[valid_idx])
                            
                            # Apply antenna phase and amplitude
                            signals[ant_idx, chirp_idx, valid_idx] += (
                                antenna_phases[ant_idx] * baseband
                            )
                            
            except Exception as e:
                logger.warning(f"Error processing scatterer: {e}")
                continue
        
        # Add noise
        noise = np.sqrt(self.noise_power) * (
            np.random.randn(*signals.shape) + 1j * np.random.randn(*signals.shape)
        )
        signals += noise
        
        return signals
    
    def process_sequence(self, 
                        sequence_path: str, 
                        output_path: str,
                        max_frames: Optional[int] = None) -> Dict:
        """
        Process an entire RadarScenes sequence.
        
        Args:
            sequence_path: Path to RadarScenes sequence directory
            output_path: Output directory for synthesized signals
            max_frames: Maximum number of frames to process (None for all)
            
        Returns:
            Processing statistics
        """
        logger.info(f"Processing sequence: {sequence_path}")
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Load sequence metadata
        metadata_path = os.path.join(sequence_path, "scenes.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load radar data
        radar_data_path = os.path.join(sequence_path, "radar_data.h5")
        if not os.path.exists(radar_data_path):
            raise FileNotFoundError(f"Radar data file not found: {radar_data_path}")
        
        # Process frames
        stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'total_scatterers': 0,
            'valid_scatterers': 0
        }
        
        with h5py.File(radar_data_path, 'r') as hdf:
            radar_data = hdf['radar_data'][:]
            
        # Group by timestamp to get frames
        timestamps = radar_data['timestamp']
        unique_timestamps = np.unique(timestamps)
            
        if max_frames:
            unique_timestamps = unique_timestamps[:max_frames]
        
        stats['total_frames'] = len(unique_timestamps)
        
        for frame_idx, timestamp in enumerate(tqdm(unique_timestamps, desc="Processing frames")):
            try:
                # Extract scatterers for this frame
                frame_mask = timestamps == timestamp
                frame_data = radar_data[frame_mask]
                
                if len(frame_data) == 0:
                    continue
                
                # Convert to DataFrame - RadarScenes uses structured arrays
                scatterers = pd.DataFrame({
                    'range_sc': frame_data['range_sc'],
                    'azimuth_sc': frame_data['azimuth_sc'],
                    'rcs': frame_data['rcs'],
                    'vr': frame_data['vr'],
                    'x_cc': frame_data['x_cc'],
                    'y_cc': frame_data['y_cc']
                })
                
                stats['total_scatterers'] += len(scatterers)
                stats['valid_scatterers'] += len(scatterers[scatterers['range_sc'] > 0])
                
                # Synthesize raw signals
                raw_signals = self.synthesize_frame(scatterers, frame_idx)
                
                # Save frame data
                frame_filename = f"frame_{frame_idx:04d}.npy"
                frame_path = os.path.join(output_path, frame_filename)
                np.save(frame_path, raw_signals)
                
                stats['processed_frames'] += 1
                
            except Exception as e:
                logger.error(f"Error processing frame {frame_idx}: {e}")
                continue
        
        # Save metadata
        metadata_filename = os.path.join(output_path, "synthesis_metadata.json")
        with open(metadata_filename, 'w') as f:
            json.dump({
                'radar_params': {
                    'fc': self.fc,
                    'bandwidth': self.bandwidth,
                    'chirp_duration': self.chirp_duration,
                    'pri': self.pri,
                    'num_chirps': self.num_chirps,
                    'num_antennas': self.num_antennas,
                    'antenna_spacing': self.antenna_spacing,
                    'sampling_rate': self.sampling_rate
                },
                'processing_stats': stats
            }, f, indent=2)
        
        logger.info(f"Processing complete:")
        logger.info(f"  Total frames: {stats['total_frames']}")
        logger.info(f"  Processed frames: {stats['processed_frames']}")
        logger.info(f"  Total scatterers: {stats['total_scatterers']}")
        logger.info(f"  Valid scatterers: {stats['valid_scatterers']}")
        
        return stats


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Synthesize raw FMCW signals from RadarScenes data')
    parser.add_argument('--seq', required=True, help='Sequence name (e.g., sequence_125)')
    parser.add_argument('--input', required=True, help='Input RadarScenes data directory')
    parser.add_argument('--out', required=True, help='Output directory for synthesized signals')
    parser.add_argument('--max-frames', type=int, help='Maximum number of frames to process')
    parser.add_argument('--fc', type=float, default=77e9, help='Carrier frequency (Hz)')
    parser.add_argument('--bandwidth', type=float, default=1e9, help='Bandwidth (Hz)')
    parser.add_argument('--chirp-duration', type=float, default=40e-6, help='Chirp duration (s)')
    parser.add_argument('--pri', type=float, default=100e-6, help='Pulse repetition interval (s)')
    parser.add_argument('--num-chirps', type=int, default=64, help='Number of chirps per frame')
    parser.add_argument('--num-antennas', type=int, default=8, help='Number of antenna elements')
    parser.add_argument('--sampling-rate', type=float, default=10e6, help='Sampling rate (Hz)')
    parser.add_argument('--noise-power', type=float, default=0.01, help='Noise power')
    
    args = parser.parse_args()
    
    # Initialize simulator
    simulator = FMCWRadarSimulator(
        fc=args.fc,
        bandwidth=args.bandwidth,
        chirp_duration=args.chirp_duration,
        pri=args.pri,
        num_chirps=args.num_chirps,
        num_antennas=args.num_antennas,
        sampling_rate=args.sampling_rate,
        noise_power=args.noise_power
    )
    
    # Process sequence
    sequence_path = os.path.join(args.input, args.seq)
    stats = simulator.process_sequence(
        sequence_path=sequence_path,
        output_path=args.out,
        max_frames=args.max_frames
    )
    
    print(f"Synthesis complete! Output saved to: {args.out}")
    print(f"Statistics: {stats}")


if __name__ == "__main__":
    main()
