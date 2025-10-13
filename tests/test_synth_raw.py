"""
Test Raw Signal Synthesis

This test verifies that the raw signal synthesis produces correct
range and Doppler peaks after dechirp and FFT operations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.simulate_raw import FMCWRadarSimulator
from src.radar_signal.dechirp import SignalPreprocessor

def test_single_target_synthesis():
    """Test synthesis with a single stationary target."""
    print("Testing single target synthesis...")
    
    # Initialize simulator
    simulator = FMCWRadarSimulator(
        fc=77e9,
        bandwidth=1e9,
        chirp_duration=40e-6,
        pri=100e-6,
        num_chirps=64,
        num_antennas=8,
        sampling_rate=10e6,
        noise_power=0.01
    )
    
    # Create single target
    scatterers = pd.DataFrame([{
        'range_sc': 50.0,  # 50 meters
        'azimuth_sc': 0.0,  # 0 degrees
        'rcs': -10.0,  # -10 dBsm
        'vr': 0.0,  # 0 m/s (stationary)
        'x_cc': 50.0,
        'y_cc': 0.0
    }])
    
    # Synthesize frame
    raw_signals = simulator.synthesize_frame(scatterers)
    
    print(f"Raw signals shape: {raw_signals.shape}")
    
    # Process with dechirp
    preprocessor = SignalPreprocessor(
        fc=77e9,
        bandwidth=1e9,
        chirp_duration=40e-6,
        pri=100e-6,
        num_chirps=64,
        sampling_rate=10e6
    )
    
    # Generate RDS
    rds = preprocessor.generate_range_doppler_spectrum(raw_signals)
    
    print(f"RDS shape: {rds.shape}")
    
    # Extract peaks
    peak_info = preprocessor.extract_range_doppler_peaks(rds, threshold_db=-30.0)
    
    print(f"Found {len(peak_info['peaks'])} peaks")
    
    # Check if we found the target
    target_found = False
    for peak in peak_info['peaks']:
        range_m = peak['range_m']
        if 45 <= range_m <= 55:  # Within 5m of expected range
            target_found = True
            print(f"Target found at range: {range_m:.2f} m")
            break
    
    assert target_found, "Target not found in expected range"
    print("✓ Single target synthesis test passed")
    
    return raw_signals, rds, peak_info

def test_moving_target_synthesis():
    """Test synthesis with a moving target."""
    print("Testing moving target synthesis...")
    
    # Initialize simulator
    simulator = FMCWRadarSimulator(
        fc=77e9,
        bandwidth=1e9,
        chirp_duration=40e-6,
        pri=100e-6,
        num_chirps=64,
        num_antennas=8,
        sampling_rate=10e6,
        noise_power=0.01
    )
    
    # Create moving target
    scatterers = pd.DataFrame([{
        'range_sc': 30.0,  # 30 meters
        'azimuth_sc': np.radians(30.0),  # 30 degrees
        'rcs': -5.0,  # -5 dBsm
        'vr': 10.0,  # 10 m/s radial velocity
        'x_cc': 30.0 * np.cos(np.radians(30.0)),
        'y_cc': 30.0 * np.sin(np.radians(30.0))
    }])
    
    # Synthesize frame
    raw_signals = simulator.synthesize_frame(scatterers)
    
    # Process with dechirp
    preprocessor = SignalPreprocessor(
        fc=77e9,
        bandwidth=1e9,
        chirp_duration=40e-6,
        pri=100e-6,
        num_chirps=64,
        sampling_rate=10e6
    )
    
    # Generate RDS
    rds = preprocessor.generate_range_doppler_spectrum(raw_signals)
    
    # Extract peaks
    peak_info = preprocessor.extract_range_doppler_peaks(rds, threshold_db=-30.0)
    
    print(f"Found {len(peak_info['peaks'])} peaks")
    
    # Check if we found the moving target
    target_found = False
    for peak in peak_info['peaks']:
        range_m = peak['range_m']
        doppler_hz = peak['doppler_hz']
        
        if 25 <= range_m <= 35 and abs(doppler_hz) > 0:  # Moving target
            target_found = True
            print(f"Moving target found at range: {range_m:.2f} m, Doppler: {doppler_hz:.2f} Hz")
            break
    
    assert target_found, "Moving target not found"
    print("✓ Moving target synthesis test passed")
    
    return raw_signals, rds, peak_info

def test_multiple_targets_synthesis():
    """Test synthesis with multiple targets."""
    print("Testing multiple targets synthesis...")
    
    # Initialize simulator
    simulator = FMCWRadarSimulator(
        fc=77e9,
        bandwidth=1e9,
        chirp_duration=40e-6,
        pri=100e-6,
        num_chirps=64,
        num_antennas=8,
        sampling_rate=10e6,
        noise_power=0.01
    )
    
    # Create multiple targets
    scatterers = pd.DataFrame([
        {
            'range_sc': 20.0,
            'azimuth_sc': 0.0,
            'rcs': -10.0,
            'vr': 0.0,
            'x_cc': 20.0,
            'y_cc': 0.0
        },
        {
            'range_sc': 40.0,
            'azimuth_sc': np.radians(45.0),
            'rcs': -8.0,
            'vr': 5.0,
            'x_cc': 40.0 * np.cos(np.radians(45.0)),
            'y_cc': 40.0 * np.sin(np.radians(45.0))
        },
        {
            'range_sc': 60.0,
            'azimuth_sc': np.radians(-30.0),
            'rcs': -12.0,
            'vr': -3.0,
            'x_cc': 60.0 * np.cos(np.radians(-30.0)),
            'y_cc': 60.0 * np.sin(np.radians(-30.0))
        }
    ])
    
    # Synthesize frame
    raw_signals = simulator.synthesize_frame(scatterers)
    
    # Process with dechirp
    preprocessor = SignalPreprocessor(
        fc=77e9,
        bandwidth=1e9,
        chirp_duration=40e-6,
        pri=100e-6,
        num_chirps=64,
        sampling_rate=10e6
    )
    
    # Generate RDS
    rds = preprocessor.generate_range_doppler_spectrum(raw_signals)
    
    # Extract peaks
    peak_info = preprocessor.extract_range_doppler_peaks(rds, threshold_db=-30.0)
    
    print(f"Found {len(peak_info['peaks'])} peaks")
    
    # Check if we found multiple targets
    target_ranges = [peak['range_m'] for peak in peak_info['peaks']]
    expected_ranges = [20.0, 40.0, 60.0]
    
    targets_found = 0
    for expected_range in expected_ranges:
        for target_range in target_ranges:
            if abs(target_range - expected_range) <= 5.0:  # Within 5m
                targets_found += 1
                break
    
    assert targets_found >= 2, f"Expected at least 2 targets, found {targets_found}"
    print(f"✓ Multiple targets synthesis test passed ({targets_found} targets found)")
    
    return raw_signals, rds, peak_info

def test_signal_processing_pipeline():
    """Test the complete signal processing pipeline."""
    print("Testing complete signal processing pipeline...")
    
    # Test single target
    raw_signals, rds, peak_info = test_single_target_synthesis()
    
    # Test moving target
    raw_signals_moving, rds_moving, peak_info_moving = test_moving_target_synthesis()
    
    # Test multiple targets
    raw_signals_multi, rds_multi, peak_info_multi = test_multiple_targets_synthesis()
    
    print("✓ Complete signal processing pipeline test passed")
    
    return {
        'single_target': (raw_signals, rds, peak_info),
        'moving_target': (raw_signals_moving, rds_moving, peak_info_moving),
        'multiple_targets': (raw_signals_multi, rds_multi, peak_info_multi)
    }

def visualize_test_results(results):
    """Visualize test results."""
    print("Visualizing test results...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Single target RDS
    ax = axes[0, 0]
    rds = results['single_target'][1]
    power_spectrum = np.abs(rds[0, :, :])**2
    power_db = 10 * np.log10(power_spectrum + 1e-12)
    im = ax.imshow(power_db, aspect='auto', origin='lower', cmap='jet')
    ax.set_title('Single Target RDS')
    ax.set_xlabel('Doppler Bin')
    ax.set_ylabel('Range Bin')
    plt.colorbar(im, ax=ax)
    
    # Moving target RDS
    ax = axes[0, 1]
    rds = results['moving_target'][1]
    power_spectrum = np.abs(rds[0, :, :])**2
    power_db = 10 * np.log10(power_spectrum + 1e-12)
    im = ax.imshow(power_db, aspect='auto', origin='lower', cmap='jet')
    ax.set_title('Moving Target RDS')
    ax.set_xlabel('Doppler Bin')
    ax.set_ylabel('Range Bin')
    plt.colorbar(im, ax=ax)
    
    # Multiple targets RDS
    ax = axes[0, 2]
    rds = results['multiple_targets'][1]
    power_spectrum = np.abs(rds[0, :, :])**2
    power_db = 10 * np.log10(power_spectrum + 1e-12)
    im = ax.imshow(power_db, aspect='auto', origin='lower', cmap='jet')
    ax.set_title('Multiple Targets RDS')
    ax.set_xlabel('Doppler Bin')
    ax.set_ylabel('Range Bin')
    plt.colorbar(im, ax=ax)
    
    # Peak detection results
    ax = axes[1, 0]
    peaks = results['single_target'][2]['peaks']
    if peaks:
        ranges = [peak['range_m'] for peak in peaks]
        powers = [peak['power_db'] for peak in peaks]
        ax.scatter(ranges, powers, s=50, alpha=0.7)
        ax.set_xlabel('Range (m)')
        ax.set_ylabel('Power (dB)')
        ax.set_title('Single Target Peaks')
        ax.grid(True)
    
    ax = axes[1, 1]
    peaks = results['moving_target'][2]['peaks']
    if peaks:
        ranges = [peak['range_m'] for peak in peaks]
        dopplers = [peak['doppler_hz'] for peak in peaks]
        ax.scatter(ranges, dopplers, s=50, alpha=0.7)
        ax.set_xlabel('Range (m)')
        ax.set_ylabel('Doppler (Hz)')
        ax.set_title('Moving Target Peaks')
        ax.grid(True)
    
    ax = axes[1, 2]
    peaks = results['multiple_targets'][2]['peaks']
    if peaks:
        ranges = [peak['range_m'] for peak in peaks]
        dopplers = [peak['doppler_hz'] for peak in peaks]
        ax.scatter(ranges, dopplers, s=50, alpha=0.7)
        ax.set_xlabel('Range (m)')
        ax.set_ylabel('Doppler (Hz)')
        ax.set_title('Multiple Targets Peaks')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('test_synthesis_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Test results visualization complete")

if __name__ == "__main__":
    print("Running raw signal synthesis tests...")
    
    try:
        # Run tests
        results = test_signal_processing_pipeline()
        
        # Visualize results
        visualize_test_results(results)
        
        print("\n✓ All tests passed successfully!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
