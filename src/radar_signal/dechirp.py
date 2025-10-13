"""
Signal Preprocessing and Dechirp Module

This module implements the signal preprocessing steps described in the paper:
1. Dechirp (subtract reference chirp)
2. Windowing and DC removal
3. 2D FFT for Range-Doppler Spectrum (RDS) generation
4. Chirp subset selection (u to u+N_L)

Following the signal model from "3-D Ego-Motion Estimation Using Multi-Channel FMCW Radar"
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class SignalPreprocessor:
    """
    Signal preprocessing for FMCW radar data.
    
    Implements dechirp, windowing, and 2D FFT operations as described
    in the paper's signal model.
    """
    
    def __init__(self,
                 fc: float = 77e9,
                 bandwidth: float = 1e9,
                 chirp_duration: float = 40e-6,
                 pri: float = 100e-6,
                 num_chirps: int = 64,
                 sampling_rate: float = 10e6,
                 window_type: str = 'hann',
                 dc_removal: bool = True):
        """
        Initialize signal preprocessor.
        
        Args:
            fc: Carrier frequency (Hz)
            bandwidth: Chirp bandwidth (Hz)
            chirp_duration: Duration of each chirp (s)
            pri: Pulse repetition interval (s)
            num_chirps: Number of chirps per frame
            sampling_rate: ADC sampling rate (Hz)
            window_type: Window function type ('hann', 'hamming', 'blackman')
            dc_removal: Whether to remove DC component
        """
        self.fc = fc
        self.bandwidth = bandwidth
        self.chirp_duration = chirp_duration
        self.pri = pri
        self.num_chirps = num_chirps
        self.sampling_rate = sampling_rate
        self.window_type = window_type
        self.dc_removal = dc_removal
        
        # Derived parameters
        self.c = 3e8  # Speed of light
        self.lambda_c = self.c / self.fc
        self.samples_per_chirp = int(self.chirp_duration * self.sampling_rate)
        self.chirp_rate = self.bandwidth / self.chirp_duration
        
        # Frequency and range resolution
        self.range_resolution = self.c / (2 * self.bandwidth)
        self.velocity_resolution = self.lambda_c / (2 * self.num_chirps * self.pri)
        
        logger.info(f"Initialized signal preprocessor:")
        logger.info(f"  Range resolution: {self.range_resolution:.2f} m")
        logger.info(f"  Velocity resolution: {self.velocity_resolution:.2f} m/s")
    
    def generate_reference_chirp(self) -> np.ndarray:
        """
        Generate the reference chirp signal for dechirping.
        
        Returns:
            Complex reference chirp signal
        """
        t = np.linspace(0, self.chirp_duration, self.samples_per_chirp)
        phase = 2 * np.pi * (self.fc * t + 0.5 * self.chirp_rate * t**2)
        return np.exp(1j * phase)
    
    def apply_window(self, signal: np.ndarray, window_type: str = None) -> np.ndarray:
        """
        Apply window function to signal.
        
        Args:
            signal: Input signal
            window_type: Window function type
            
        Returns:
            Windowed signal
        """
        if window_type is None:
            window_type = self.window_type
        
        if window_type == 'hann':
            window = windows.hann(len(signal))
        elif window_type == 'hamming':
            window = windows.hamming(len(signal))
        elif window_type == 'blackman':
            window = windows.blackman(len(signal))
        else:
            raise ValueError(f"Unknown window type: {window_type}")
        
        return signal * window
    
    def remove_dc(self, signal: np.ndarray) -> np.ndarray:
        """
        Remove DC component from signal.
        
        Args:
            signal: Input signal
            
        Returns:
            DC-removed signal
        """
        return signal - np.mean(signal)
    
    def dechirp_signal(self, 
                      received_signal: np.ndarray,
                      reference_chirp: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Perform dechirp operation (mix received signal with reference).
        
        Args:
            received_signal: Received complex signal [samples]
            reference_chirp: Reference chirp signal (generated if None)
            
        Returns:
            Dechirped baseband signal
        """
        if reference_chirp is None:
            reference_chirp = self.generate_reference_chirp()
        
        # Mix received signal with conjugate of reference chirp
        baseband = received_signal * np.conj(reference_chirp)
        
        return baseband
    
    def process_chirp(self, 
                     chirp_signal: np.ndarray,
                     reference_chirp: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Process a single chirp: dechirp, window, remove DC.
        
        Args:
            chirp_signal: Input chirp signal [samples]
            reference_chirp: Reference chirp (generated if None)
            
        Returns:
            Processed baseband signal
        """
        # Dechirp
        baseband = self.dechirp_signal(chirp_signal, reference_chirp)
        
        # Apply window
        baseband = self.apply_window(baseband)
        
        # Remove DC if enabled
        if self.dc_removal:
            baseband = self.remove_dc(baseband)
        
        return baseband
    
    def generate_range_doppler_spectrum(self, 
                                      frame_signals: np.ndarray,
                                      chirp_subset: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Generate Range-Doppler Spectrum (RDS) from frame signals.
        
        Args:
            frame_signals: Complex signals [num_antennas, num_chirps, samples_per_chirp]
            chirp_subset: Tuple of (start_chirp, end_chirp) for subset selection
            
        Returns:
            RDS matrices [num_antennas, range_bins, doppler_bins]
        """
        num_antennas, num_chirps, samples_per_chirp = frame_signals.shape
        
        # Select chirp subset if specified
        if chirp_subset is not None:
            start_chirp, end_chirp = chirp_subset
            frame_signals = frame_signals[:, start_chirp:end_chirp, :]
            num_chirps = end_chirp - start_chirp
        
        # Generate reference chirp
        reference_chirp = self.generate_reference_chirp()
        
        # Initialize RDS array
        rds = np.zeros((num_antennas, samples_per_chirp, num_chirps), dtype=complex)
        
        # Process each antenna
        for ant_idx in range(num_antennas):
            # Process each chirp
            for chirp_idx in range(num_chirps):
                chirp_signal = frame_signals[ant_idx, chirp_idx, :]
                
                # Process chirp
                processed_chirp = self.process_chirp(chirp_signal, reference_chirp)
                
                # Store processed signal
                rds[ant_idx, :, chirp_idx] = processed_chirp
        
        # Apply 2D FFT: range (fast-time) and Doppler (slow-time)
        rds_fft = np.fft.fft2(rds, axes=(1, 2))
        
        # Shift to center zero frequency
        rds_fft = np.fft.fftshift(rds_fft, axes=(1, 2))
        
        return rds_fft
    
    def extract_range_doppler_peaks(self, 
                                   rds: np.ndarray,
                                   threshold_db: float = -20.0,
                                   min_range: float = 1.0,
                                   max_range: float = 200.0) -> Dict:
        """
        Extract peaks from Range-Doppler Spectrum.
        
        Args:
            rds: RDS matrix [num_antennas, range_bins, doppler_bins]
            threshold_db: Detection threshold in dB
            min_range: Minimum range for detection (m)
            max_range: Maximum range for detection (m)
            
        Returns:
            Dictionary with peak information
        """
        num_antennas, range_bins, doppler_bins = rds.shape
        
        # Convert to power spectrum
        power_spectrum = np.abs(rds)**2
        
        # Convert to dB
        power_db = 10 * np.log10(power_spectrum + 1e-12)
        
        # Range and Doppler bin indices
        range_bins_m = np.linspace(0, self.range_resolution * range_bins, range_bins)
        doppler_bins_hz = np.linspace(-self.sampling_rate/2, self.sampling_rate/2, doppler_bins)
        
        # Find peaks above threshold
        peaks = []
        for ant_idx in range(num_antennas):
            ant_spectrum = power_db[ant_idx, :, :]
            
            # Find local maxima above threshold
            from scipy.ndimage import maximum_filter
            local_maxima = maximum_filter(ant_spectrum, size=3) == ant_spectrum
            above_threshold = ant_spectrum > threshold_db
            
            peak_mask = local_maxima & above_threshold
            
            # Extract peak coordinates
            peak_indices = np.where(peak_mask)
            for i, j in zip(peak_indices[0], peak_indices[1]):
                range_val = range_bins_m[i]
                doppler_val = doppler_bins_hz[j]
                
                # Check range constraints
                if min_range <= range_val <= max_range:
                    peaks.append({
                        'antenna': ant_idx,
                        'range_bin': i,
                        'doppler_bin': j,
                        'range_m': range_val,
                        'doppler_hz': doppler_val,
                        'power_db': ant_spectrum[i, j]
                    })
        
        return {
            'peaks': peaks,
            'range_bins_m': range_bins_m,
            'doppler_bins_hz': doppler_bins_hz,
            'power_spectrum_db': power_db
        }
    
    def visualize_rds(self, 
                     rds: np.ndarray,
                     antenna_idx: int = 0,
                     save_path: Optional[str] = None) -> None:
        """
        Visualize Range-Doppler Spectrum.
        
        Args:
            rds: RDS matrix [num_antennas, range_bins, doppler_bins]
            antenna_idx: Antenna index to visualize
            save_path: Path to save plot
        """
        power_spectrum = np.abs(rds[antenna_idx, :, :])**2
        power_db = 10 * np.log10(power_spectrum + 1e-12)
        
        # Create range and Doppler axes
        range_bins = np.linspace(0, self.range_resolution * rds.shape[1], rds.shape[1])
        doppler_bins = np.linspace(-self.sampling_rate/2, self.sampling_rate/2, rds.shape[2])
        
        plt.figure(figsize=(10, 6))
        plt.imshow(power_db, aspect='auto', origin='lower', 
                  extent=[doppler_bins[0], doppler_bins[-1], range_bins[0], range_bins[-1]],
                  cmap='jet')
        plt.colorbar(label='Power (dB)')
        plt.xlabel('Doppler Frequency (Hz)')
        plt.ylabel('Range (m)')
        plt.title(f'Range-Doppler Spectrum (Antenna {antenna_idx})')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def process_frame(raw_signals_path: str, 
                 output_path: str,
                 radar_params: Dict,
                 chirp_subset: Optional[Tuple[int, int]] = None) -> Dict:
    """
    Process a single frame of raw signals.
    
    Args:
        raw_signals_path: Path to raw signals file (.npy)
        output_path: Path to save processed RDS
        radar_params: Radar parameters dictionary
        chirp_subset: Chirp subset selection (start, end)
        
    Returns:
        Processing results dictionary
    """
    # Load raw signals
    raw_signals = np.load(raw_signals_path)
    logger.info(f"Loaded raw signals: {raw_signals.shape}")
    
    # Initialize preprocessor
    preprocessor = SignalPreprocessor(**radar_params)
    
    # Generate RDS
    rds = preprocessor.generate_range_doppler_spectrum(raw_signals, chirp_subset)
    logger.info(f"Generated RDS: {rds.shape}")
    
    # Extract peaks
    peak_info = preprocessor.extract_range_doppler_peaks(rds)
    logger.info(f"Found {len(peak_info['peaks'])} peaks")
    
    # Save results
    np.save(output_path, rds)
    
    # Save peak information
    peak_path = output_path.replace('.npy', '_peaks.npz')
    np.savez(peak_path, **peak_info)
    
    return {
        'rds_shape': rds.shape,
        'num_peaks': len(peak_info['peaks']),
        'peak_info': peak_info
    }


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Process raw FMCW signals')
    parser.add_argument('--raw', required=True, help='Path to raw signals file')
    parser.add_argument('--out', required=True, help='Output path for RDS')
    parser.add_argument('--chirp-start', type=int, help='Start chirp index')
    parser.add_argument('--chirp-end', type=int, help='End chirp index')
    
    args = parser.parse_args()
    
    # Default radar parameters
    radar_params = {
        'fc': 77e9,
        'bandwidth': 1e9,
        'chirp_duration': 40e-6,
        'pri': 100e-6,
        'num_chirps': 64,
        'sampling_rate': 10e6
    }
    
    # Chirp subset
    chirp_subset = None
    if args.chirp_start is not None and args.chirp_end is not None:
        chirp_subset = (args.chirp_start, args.chirp_end)
    
    # Process frame
    results = process_frame(args.raw, args.out, radar_params, chirp_subset)
    print(f"Processing complete: {results}")
