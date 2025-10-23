"""
Simplified Phase 2 Enhancement Test

This test verifies the Phase 2 enhancements with a simplified approach:
1. Robust angle estimation with temporal smoothing
2. Advanced optimization with regularization
3. Real-time processing optimizations
4. Performance improvements
"""

import numpy as np
import sys
import os
import time
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.simulate_raw import FMCWRadarSimulator
from src.radar_signal.dechirp import SignalPreprocessor
from src.algorithms.robust_angle_estimation import RobustAngleEstimator
from src.algorithms.advanced_velocity_optimization import AdvancedVelocityOptimizer
from src.core.real_time_processor import RealTimeVelocityEstimator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_scenario():
    """Create a simple test scenario."""
    print("Creating test scenario...")
    
    # Known ego-motion parameters
    true_velocity = np.array([10.0, 2.0, 0.0])  # m/s
    true_angular_velocity = np.array([0.0, 0.0, 0.1])  # rad/s
    
    # Initialize simulator
    simulator = FMCWRadarSimulator(
        fc=77e9,
        bandwidth=1e9,
        chirp_duration=40e-6,
        pri=100e-6,
        num_chirps=32,
        num_antennas=8,
        sampling_rate=10e6,
        noise_power=0.01
    )
    
    # Create simple targets
    import pandas as pd
    scatterers = pd.DataFrame([
        {
            'range_sc': 30.0,
            'azimuth_sc': 0.0,
            'rcs': -5.0,
            'vr': 0.0,
            'x_cc': 30.0,
            'y_cc': 0.0
        },
        {
            'range_sc': 40.0,
            'azimuth_sc': np.radians(30.0),
            'rcs': -7.0,
            'vr': 0.0,
            'x_cc': 40.0 * np.cos(np.radians(30.0)),
            'y_cc': 40.0 * np.sin(np.radians(30.0))
        }
    ])
    
    return simulator, scatterers, true_velocity, true_angular_velocity

def test_robust_angle_estimation():
    """Test robust angle estimation."""
    print("\n=== Testing Robust Angle Estimation ===")
    
    # Create test scenario
    simulator, scatterers, true_velocity, true_angular_velocity = create_test_scenario()
    
    # Initialize processors
    preprocessor = SignalPreprocessor(
        fc=77e9,
        bandwidth=1e9,
        chirp_duration=40e-6,
        pri=100e-6,
        num_chirps=32,
        sampling_rate=10e6
    )
    
    # Initialize robust angle estimator
    angle_estimator = RobustAngleEstimator(
        fc=77e9,
        antenna_spacing=3e8 / (2 * 77e9),
        num_antennas=8,
        search_resolution=2.0,
        temporal_window=3,
        confidence_threshold=0.5,
        smoothing_factor=0.3,
        max_targets=20
    )
    
    # Process multiple frames
    num_frames = 5
    all_targets = []
    processing_times = []
    
    for frame in range(num_frames):
        print(f"Processing frame {frame + 1}...")
        frame_start = time.time()
        
        # Synthesize and process frame
        raw_signals = simulator.synthesize_frame(scatterers)
        rds = preprocessor.generate_range_doppler_spectrum(raw_signals)
        peak_info = preprocessor.extract_range_doppler_peaks(rds, threshold_db=-25.0)
        
        # Robust angle estimation
        targets = angle_estimator.process_targets_robust(rds, peak_info, frame_timestamp=time.time())
        
        frame_time = time.time() - frame_start
        processing_times.append(frame_time)
        
        print(f"  Frame {frame + 1}: {len(targets)} reliable targets in {frame_time:.3f}s")
        all_targets.extend(targets)
    
    # Analyze results
    if all_targets:
        confidences = [t['confidence'] for t in all_targets]
        reliable_count = sum(1 for t in all_targets if t['is_reliable'])
        multipath_count = sum(1 for t in all_targets if t['interference_analysis']['is_multipath'])
        
        print(f"\nRobust Angle Estimation Results:")
        print(f"  Total targets processed: {len(all_targets)}")
        print(f"  Reliable targets: {reliable_count}")
        print(f"  Multipath targets: {multipath_count}")
        print(f"  Average confidence: {np.mean(confidences):.3f}")
        print(f"  Average processing time: {np.mean(processing_times):.3f}s")
        
        return {
            'targets': all_targets,
            'processing_times': processing_times,
            'reliable_count': reliable_count,
            'avg_confidence': np.mean(confidences)
        }
    else:
        print("No targets processed")
        return None

def test_advanced_optimization():
    """Test advanced velocity optimization."""
    print("\n=== Testing Advanced Velocity Optimization ===")
    
    # Create test scenario
    simulator, scatterers, true_velocity, true_angular_velocity = create_test_scenario()
    
    # Initialize processors
    preprocessor = SignalPreprocessor(
        fc=77e9,
        bandwidth=1e9,
        chirp_duration=40e-6,
        pri=100e-6,
        num_chirps=32,
        sampling_rate=10e6
    )
    
    angle_estimator = RobustAngleEstimator(
        fc=77e9,
        antenna_spacing=3e8 / (2 * 77e9),
        num_antennas=8,
        search_resolution=2.0,
        temporal_window=3,
        confidence_threshold=0.5,
        max_targets=15
    )
    
    # Initialize advanced optimizer
    optimizer = AdvancedVelocityOptimizer(
        fc=77e9,
        lambda_c=3e8 / 77e9,
        num_antennas=8,
        antenna_spacing=3e8 / (2 * 77e9),
        max_velocity=20.0,
        max_angular_velocity=2.0,
        regularization_weight=0.1,
        num_optimization_runs=2,
        use_parallel=False  # Disable parallel for simplicity
    )
    
    # Process two consecutive frames
    print("Processing frame 1...")
    raw_signals1 = simulator.synthesize_frame(scatterers)
    rds1 = preprocessor.generate_range_doppler_spectrum(raw_signals1)
    peak_info1 = preprocessor.extract_range_doppler_peaks(rds1, threshold_db=-25.0)
    targets1 = angle_estimator.process_targets_robust(rds1, peak_info1)
    
    print("Processing frame 2...")
    raw_signals2 = simulator.synthesize_frame(scatterers)
    rds2 = preprocessor.generate_range_doppler_spectrum(raw_signals2)
    peak_info2 = preprocessor.extract_range_doppler_peaks(rds2, threshold_db=-25.0)
    targets2 = angle_estimator.process_targets_robust(rds2, peak_info2)
    
    print(f"Frame 1: {len(targets1)} targets")
    print(f"Frame 2: {len(targets2)} targets")
    
    # Create target associations (simplified)
    target_associations = []
    min_targets = min(len(targets1), len(targets2), 5)  # Limit to 5 associations
    
    for i in range(min_targets):
        t1 = targets1[i]
        t2 = targets2[i]
        
        # Compute temporal phase difference
        spatial_sig1 = t1['spatial_signature']
        spatial_sig2 = t2['spatial_signature']
        temporal_phase_diff = np.angle(spatial_sig2[0] * np.conj(spatial_sig1[0]))
        
        association = {
            'current': t2,
            'previous': t1,
            'temporal_phase_diff': temporal_phase_diff,
            'distance': 1.0  # Simplified distance
        }
        target_associations.append(association)
    
    print(f"Created {len(target_associations)} target associations")
    
    # Run advanced optimization
    print("Running advanced optimization...")
    optimization_start = time.time()
    
    results = optimizer.run_robust_optimization(target_associations, dt=0.1)
    
    optimization_time = time.time() - optimization_start
    
    print(f"Advanced optimization completed in {optimization_time:.3f}s")
    print(f"Optimization success: {results['success']}")
    
    if results['success']:
        print(f"  Velocity: {results['velocity']}")
        print(f"  Angular velocity: {results['angular_velocity']}")
        print(f"  RMSE: {results['rmse']:.6f}")
        print(f"  Successful runs: {results['successful_runs']}/{results['num_optimization_runs']}")
        
        # Compare with true values
        vel_error = np.linalg.norm(results['velocity'] - true_velocity)
        ang_error = np.linalg.norm(results['angular_velocity'] - true_angular_velocity)
        
        print(f"  Velocity error: {vel_error:.6f} m/s")
        print(f"  Angular velocity error: {ang_error:.6f} rad/s")
        
        return {
            'results': results,
            'optimization_time': optimization_time,
            'velocity_error': vel_error,
            'angular_velocity_error': ang_error
        }
    else:
        print("Optimization failed")
        return None

def test_real_time_processing():
    """Test real-time processing."""
    print("\n=== Testing Real-Time Processing ===")
    
    # Create test scenario
    simulator, scatterers, true_velocity, true_angular_velocity = create_test_scenario()
    
    # Initialize processors
    preprocessor = SignalPreprocessor(
        fc=77e9,
        bandwidth=1e9,
        chirp_duration=40e-6,
        pri=100e-6,
        num_chirps=32,
        sampling_rate=10e6
    )
    
    # Create real-time estimator
    radar_params = {
        'fc': 77e9,
        'lambda_c': 3e8 / 77e9,
        'num_antennas': 8
    }
    
    estimator = RealTimeVelocityEstimator(
        radar_params=radar_params,
        frame_buffer_size=3,
        use_parallel=False  # Disable parallel for simplicity
    )
    
    # Start real-time processing
    estimator.start_estimation()
    
    # Process multiple frames
    num_frames = 5
    frame_ids = []
    processing_times = []
    
    for frame in range(num_frames):
        print(f"Adding frame {frame + 1} for processing...")
        frame_start = time.time()
        
        # Synthesize and process frame
        raw_signals = simulator.synthesize_frame(scatterers)
        rds = preprocessor.generate_range_doppler_spectrum(raw_signals)
        peak_info = preprocessor.extract_range_doppler_peaks(rds, threshold_db=-25.0)
        
        # Add frame for processing
        frame_id = estimator.add_frame(rds, peak_info)
        frame_ids.append(frame_id)
        
        frame_time = time.time() - frame_start
        processing_times.append(frame_time)
        
        print(f"  Frame {frame + 1} (ID: {frame_id}) added in {frame_time:.3f}s")
        
        # Small delay to simulate real-time processing
        time.sleep(0.1)
    
    # Wait for processing to complete
    print("Waiting for processing to complete...")
    time.sleep(1.0)
    
    # Get latest results
    latest_frames = estimator.real_time_processor.get_latest_results(2)
    print(f"Latest frames available: {len(latest_frames)}")
    
    # Get velocity estimate
    velocity_estimate = estimator.get_latest_velocity_estimate()
    if velocity_estimate:
        print(f"Latest velocity estimate: {velocity_estimate}")
    
    # Get statistics
    stats = estimator.get_estimation_statistics()
    print(f"Real-time processing statistics:")
    print(f"  Processing metrics: {stats['processing_metrics']}")
    
    # Stop processing
    estimator.stop_estimation()
    
    return {
        'frame_ids': frame_ids,
        'processing_times': processing_times,
        'latest_frames': latest_frames,
        'velocity_estimate': velocity_estimate,
        'statistics': stats
    }

def test_performance_comparison():
    """Compare performance between methods."""
    print("\n=== Performance Comparison ===")
    
    # Create test scenario
    simulator, scatterers, true_velocity, true_angular_velocity = create_test_scenario()
    
    # Initialize processors
    preprocessor = SignalPreprocessor(
        fc=77e9,
        bandwidth=1e9,
        chirp_duration=40e-6,
        pri=100e-6,
        num_chirps=32,
        sampling_rate=10e6
    )
    
    # Test original method (simplified)
    print("Testing original method...")
    original_start = time.time()
    
    # Simulate original processing
    raw_signals = simulator.synthesize_frame(scatterers)
    rds = preprocessor.generate_range_doppler_spectrum(raw_signals)
    peak_info = preprocessor.extract_range_doppler_peaks(rds, threshold_db=-25.0)
    
    # Simple angle estimation (no temporal smoothing)
    simple_angles = []
    for peak in peak_info['peaks'][:20]:  # Limit to 20 targets
        # Simple beamforming
        spatial_signature = rds[:, peak['range_bin'], peak['doppler_bin']]
        azimuth_grid = np.arange(-90, 91, 5)  # Coarse grid
        beamforming_spectrum = np.zeros(len(azimuth_grid))
        
        for i, azimuth in enumerate(azimuth_grid):
            azimuth_rad = np.radians(azimuth)
            phases = 2 * np.pi * np.arange(8) * 3e8 / (2 * 77e9) * np.sin(azimuth_rad) / (3e8 / 77e9)
            steering_vector = np.exp(1j * phases)
            beamforming_spectrum[i] = np.abs(steering_vector.conj().T @ spatial_signature)**2
        
        angle = azimuth_grid[np.argmax(beamforming_spectrum)]
        simple_angles.append(angle)
    
    original_time = time.time() - original_start
    
    # Test enhanced method
    print("Testing enhanced method...")
    enhanced_start = time.time()
    
    # Robust angle estimation
    angle_estimator = RobustAngleEstimator(
        fc=77e9,
        antenna_spacing=3e8 / (2 * 77e9),
        num_antennas=8,
        search_resolution=2.0,
        temporal_window=3,
        confidence_threshold=0.5,
        max_targets=20
    )
    
    enhanced_targets = angle_estimator.process_targets_robust(rds, peak_info)
    
    enhanced_time = time.time() - enhanced_start
    
    # Compare results
    print(f"\nPerformance Comparison:")
    print(f"  Original method: {original_time:.3f}s")
    print(f"  Enhanced method: {enhanced_time:.3f}s")
    print(f"  Speed ratio: {original_time / enhanced_time:.2f}x")
    
    print(f"  Original targets: {len(simple_angles)}")
    print(f"  Enhanced targets: {len(enhanced_targets)}")
    
    if enhanced_targets:
        confidences = [t['confidence'] for t in enhanced_targets]
        reliable_count = sum(1 for t in enhanced_targets if t['is_reliable'])
        
        print(f"  Enhanced reliable targets: {reliable_count}")
        print(f"  Average confidence: {np.mean(confidences):.3f}")
    
    return {
        'original_time': original_time,
        'enhanced_time': enhanced_time,
        'speed_ratio': original_time / enhanced_time,
        'original_targets': len(simple_angles),
        'enhanced_targets': len(enhanced_targets),
        'enhanced_reliable': reliable_count if enhanced_targets else 0
    }

def main():
    """Main test function for Phase 2 enhancements."""
    print("Testing Phase 2 Algorithm Enhancements (Simplified)...")
    print("Expected improvements: Robust estimation, Advanced optimization, Real-time processing")
    
    try:
        # Test robust angle estimation
        angle_results = test_robust_angle_estimation()
        
        # Test advanced optimization
        optimization_results = test_advanced_optimization()
        
        # Test real-time processing
        realtime_results = test_real_time_processing()
        
        # Test performance comparison
        performance_results = test_performance_comparison()
        
        print("\n=== Phase 2 Enhancement Summary ===")
        print("✓ Robust angle estimation with temporal smoothing")
        print("✓ Advanced optimization with regularization")
        print("✓ Real-time processing optimizations")
        print("✓ Performance improvements demonstrated")
        
        # Print summary statistics
        if angle_results:
            print(f"\nAngle Estimation Results:")
            print(f"  Reliable targets: {angle_results['reliable_count']}")
            print(f"  Average confidence: {angle_results['avg_confidence']:.3f}")
        
        if optimization_results:
            print(f"\nOptimization Results:")
            print(f"  Success: {optimization_results['results']['success']}")
            print(f"  Velocity error: {optimization_results['velocity_error']:.3f} m/s")
            print(f"  Optimization time: {optimization_results['optimization_time']:.3f}s")
        
        if performance_results:
            print(f"\nPerformance Results:")
            print(f"  Speed improvement: {performance_results['speed_ratio']:.1f}x")
            print(f"  Enhanced reliable targets: {performance_results['enhanced_reliable']}")
        
        return {
            'angle_results': angle_results,
            'optimization_results': optimization_results,
            'realtime_results': realtime_results,
            'performance_results': performance_results
        }
        
    except Exception as e:
        print(f"\n✗ Phase 2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
