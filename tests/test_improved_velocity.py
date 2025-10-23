"""
Test Improved Velocity Estimation

This test verifies that the improved velocity estimation with proper temporal
phase differences and target association produces better results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.simulate_raw import FMCWRadarSimulator
from src.radar_signal.dechirp import SignalPreprocessor
from src.angle_estimation.angle_estimation import AngleEstimator
from src.velocity_solver.velocity_solver import VelocitySolver
from src.algorithms.velocity_solver_improved import ImprovedVelocitySolver
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_scenario():
    """Create a test scenario with known ego-motion."""
    print("Creating test scenario with known ego-motion...")
    
    # Known ego-motion parameters
    true_velocity = np.array([10.0, 2.0, 0.0])  # m/s
    true_angular_velocity = np.array([0.0, 0.0, 0.1])  # rad/s
    
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
    
    # Create multiple targets at different positions
    scatterers = pd.DataFrame([
        {
            'range_sc': 30.0,
            'azimuth_sc': 0.0,
            'rcs': -10.0,
            'vr': 0.0,
            'x_cc': 30.0,
            'y_cc': 0.0
        },
        {
            'range_sc': 50.0,
            'azimuth_sc': np.radians(30.0),
            'rcs': -8.0,
            'vr': 0.0,
            'x_cc': 50.0 * np.cos(np.radians(30.0)),
            'y_cc': 50.0 * np.sin(np.radians(30.0))
        },
        {
            'range_sc': 40.0,
            'azimuth_sc': np.radians(-45.0),
            'rcs': -12.0,
            'vr': 0.0,
            'x_cc': 40.0 * np.cos(np.radians(-45.0)),
            'y_cc': 40.0 * np.sin(np.radians(-45.0))
        }
    ])
    
    return simulator, scatterers, true_velocity, true_angular_velocity

def process_frame(simulator, scatterers, preprocessor, angle_estimator):
    """Process a single frame and return targets with angles."""
    # Synthesize raw signals
    raw_signals = simulator.synthesize_frame(scatterers)
    
    # Generate RDS
    rds = preprocessor.generate_range_doppler_spectrum(raw_signals)
    
    # Extract peaks
    peak_info = preprocessor.extract_range_doppler_peaks(rds, threshold_db=-30.0)
    
    # Extract angles
    targets = angle_estimator.process_targets(rds, peak_info, method='music')
    
    return targets, rds

def test_original_vs_improved():
    """Compare original vs improved velocity estimation."""
    print("Testing original vs improved velocity estimation...")
    
    # Create test scenario
    simulator, scatterers, true_velocity, true_angular_velocity = create_test_scenario()
    
    # Initialize processors
    preprocessor = SignalPreprocessor(
        fc=77e9,
        bandwidth=1e9,
        chirp_duration=40e-6,
        pri=100e-6,
        num_chirps=64,
        sampling_rate=10e6
    )
    
    angle_estimator = AngleEstimator(
        fc=77e9,
        antenna_spacing=3e8 / (2 * 77e9),
        num_antennas=8
    )
    
    # Process two consecutive frames
    print("Processing frame 1...")
    targets1, rds1 = process_frame(simulator, scatterers, preprocessor, angle_estimator)
    
    print("Processing frame 2...")
    targets2, rds2 = process_frame(simulator, scatterers, preprocessor, angle_estimator)
    
    print(f"Frame 1: {len(targets1)} targets")
    print(f"Frame 2: {len(targets2)} targets")
    
    # Test original velocity solver
    print("\n=== Testing Original Velocity Solver ===")
    original_solver = VelocitySolver(
        fc=77e9,
        lambda_c=3e8 / 77e9,
        num_antennas=8,
        antenna_spacing=3e8 / (2 * 77e9)
    )
    
    original_results = original_solver.solve_velocity(rds2, targets2, dt=0.1)
    
    print(f"Original solver success: {original_results['success']}")
    if original_results['success']:
        print(f"Original velocity: {original_results['velocity']}")
        print(f"Original angular velocity: {original_results['angular_velocity']}")
        print(f"Original RMSE: {original_results['rmse']:.6f}")
    
    # Test improved velocity solver
    print("\n=== Testing Improved Velocity Solver ===")
    improved_solver = ImprovedVelocitySolver(
        fc=77e9,
        lambda_c=3e8 / 77e9,
        num_antennas=8,
        antenna_spacing=3e8 / (2 * 77e9),
        association_threshold=5.0
    )
    
    improved_results = improved_solver.solve_velocity_with_association(
        targets2, targets1, dt=0.1
    )
    
    print(f"Improved solver success: {improved_results['success']}")
    if improved_results['success']:
        print(f"Improved velocity: {improved_results['velocity']}")
        print(f"Improved angular velocity: {improved_results['angular_velocity']}")
        print(f"Improved RMSE: {improved_results['rmse']:.6f}")
        print(f"Number of associations: {improved_results['num_associations']}")
    
    # Compare results
    print("\n=== Comparison ===")
    print(f"True velocity: {true_velocity}")
    print(f"True angular velocity: {true_angular_velocity}")
    
    if original_results['success'] and improved_results['success']:
        # Velocity accuracy
        original_vel_error = np.linalg.norm(original_results['velocity'] - true_velocity)
        improved_vel_error = np.linalg.norm(improved_results['velocity'] - true_velocity)
        
        print(f"\nVelocity Error:")
        print(f"  Original: {original_vel_error:.6f} m/s")
        print(f"  Improved: {improved_vel_error:.6f} m/s")
        print(f"  Improvement: {((original_vel_error - improved_vel_error) / original_vel_error * 100):.1f}%")
        
        # Angular velocity accuracy
        original_ang_error = np.linalg.norm(original_results['angular_velocity'] - true_angular_velocity)
        improved_ang_error = np.linalg.norm(improved_results['angular_velocity'] - true_angular_velocity)
        
        print(f"\nAngular Velocity Error:")
        print(f"  Original: {original_ang_error:.6f} rad/s")
        print(f"  Improved: {improved_ang_error:.6f} rad/s")
        print(f"  Improvement: {((original_ang_error - improved_ang_error) / original_ang_error * 100):.1f}%")
        
        # RMSE comparison
        print(f"\nRMSE:")
        print(f"  Original: {original_results['rmse']:.6f}")
        print(f"  Improved: {improved_results['rmse']:.6f}")
        print(f"  Improvement: {((original_results['rmse'] - improved_results['rmse']) / original_results['rmse'] * 100):.1f}%")
    
    return {
        'original': original_results,
        'improved': improved_results,
        'true_velocity': true_velocity,
        'true_angular_velocity': true_angular_velocity
    }

def test_multiple_frames():
    """Test with multiple frames to show temporal consistency."""
    print("\n=== Testing Multiple Frames ===")
    
    # Create test scenario
    simulator, scatterers, true_velocity, true_angular_velocity = create_test_scenario()
    
    # Initialize processors
    preprocessor = SignalPreprocessor(
        fc=77e9,
        bandwidth=1e9,
        chirp_duration=40e-6,
        pri=100e-6,
        num_chirps=64,
        sampling_rate=10e6
    )
    
    angle_estimator = AngleEstimator(
        fc=77e9,
        antenna_spacing=3e8 / (2 * 77e9),
        num_antennas=8
    )
    
    improved_solver = ImprovedVelocitySolver(
        fc=77e9,
        lambda_c=3e8 / 77e9,
        num_antennas=8,
        antenna_spacing=3e8 / (2 * 77e9),
        association_threshold=5.0
    )
    
    # Process multiple frames
    num_frames = 5
    velocity_estimates = []
    angular_velocity_estimates = []
    rmse_values = []
    
    previous_targets = None
    
    for frame in range(num_frames):
        print(f"Processing frame {frame + 1}...")
        
        # Process current frame
        current_targets, rds = process_frame(simulator, scatterers, preprocessor, angle_estimator)
        
        if previous_targets is not None:
            # Estimate velocity using improved method
            results = improved_solver.solve_velocity_with_association(
                current_targets, previous_targets, dt=0.1
            )
            
            if results['success']:
                velocity_estimates.append(results['velocity'])
                angular_velocity_estimates.append(results['angular_velocity'])
                rmse_values.append(results['rmse'])
                print(f"  Velocity: {results['velocity']}")
                print(f"  RMSE: {results['rmse']:.6f}")
            else:
                print(f"  Failed to estimate velocity")
        
        previous_targets = current_targets
    
    # Analyze temporal consistency
    if velocity_estimates:
        velocity_estimates = np.array(velocity_estimates)
        angular_velocity_estimates = np.array(angular_velocity_estimates)
        rmse_values = np.array(rmse_values)
        
        print(f"\nTemporal Analysis:")
        print(f"  Mean velocity: {np.mean(velocity_estimates, axis=0)}")
        print(f"  Velocity std: {np.std(velocity_estimates, axis=0)}")
        print(f"  Mean RMSE: {np.mean(rmse_values):.6f}")
        print(f"  RMSE std: {np.std(rmse_values):.6f}")
        
        # Compare with true values
        vel_error = np.linalg.norm(np.mean(velocity_estimates, axis=0) - true_velocity)
        ang_error = np.linalg.norm(np.mean(angular_velocity_estimates, axis=0) - true_angular_velocity)
        
        print(f"  Velocity error: {vel_error:.6f} m/s")
        print(f"  Angular velocity error: {ang_error:.6f} rad/s")
    
    return {
        'velocity_estimates': velocity_estimates,
        'angular_velocity_estimates': angular_velocity_estimates,
        'rmse_values': rmse_values
    }

def visualize_comparison(results):
    """Visualize comparison between original and improved methods."""
    print("Creating comparison visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Velocity comparison
    ax = axes[0, 0]
    if results['original']['success'] and results['improved']['success']:
        original_vel = results['original']['velocity']
        improved_vel = results['improved']['velocity']
        true_vel = results['true_velocity']
        
        components = ['vx', 'vy', 'vz']
        x = np.arange(len(components))
        width = 0.25
        
        ax.bar(x - width, original_vel, width, label='Original', alpha=0.7)
        ax.bar(x, improved_vel, width, label='Improved', alpha=0.7)
        ax.bar(x + width, true_vel, width, label='True', alpha=0.7)
        
        ax.set_xlabel('Velocity Components')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title('Velocity Estimation Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(components)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Angular velocity comparison
    ax = axes[0, 1]
    if results['original']['success'] and results['improved']['success']:
        original_ang = results['original']['angular_velocity']
        improved_ang = results['improved']['angular_velocity']
        true_ang = results['true_angular_velocity']
        
        components = ['wx', 'wy', 'wz']
        x = np.arange(len(components))
        width = 0.25
        
        ax.bar(x - width, original_ang, width, label='Original', alpha=0.7)
        ax.bar(x, improved_ang, width, label='Improved', alpha=0.7)
        ax.bar(x + width, true_ang, width, label='True', alpha=0.7)
        
        ax.set_xlabel('Angular Velocity Components')
        ax.set_ylabel('Angular Velocity (rad/s)')
        ax.set_title('Angular Velocity Estimation Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(components)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # RMSE comparison
    ax = axes[1, 0]
    if results['original']['success'] and results['improved']['success']:
        original_rmse = results['original']['rmse']
        improved_rmse = results['improved']['rmse']
        
        methods = ['Original', 'Improved']
        rmse_values = [original_rmse, improved_rmse]
        
        bars = ax.bar(methods, rmse_values, alpha=0.7, color=['red', 'green'])
        ax.set_ylabel('RMSE')
        ax.set_title('RMSE Comparison')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, rmse_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{value:.4f}', ha='center', va='bottom')
    
    # Error comparison
    ax = axes[1, 1]
    if results['original']['success'] and results['improved']['success']:
        original_vel_error = np.linalg.norm(results['original']['velocity'] - results['true_velocity'])
        improved_vel_error = np.linalg.norm(results['improved']['velocity'] - results['true_velocity'])
        
        methods = ['Original', 'Improved']
        error_values = [original_vel_error, improved_vel_error]
        
        bars = ax.bar(methods, error_values, alpha=0.7, color=['red', 'green'])
        ax.set_ylabel('Velocity Error (m/s)')
        ax.set_title('Velocity Error Comparison')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, error_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('improved_velocity_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Comparison visualization saved as 'improved_velocity_comparison.png'")

def main():
    """Main test function."""
    print("Running improved velocity estimation tests...")
    
    try:
        # Test original vs improved
        results = test_original_vs_improved()
        
        # Test multiple frames
        multi_frame_results = test_multiple_frames()
        
        # Visualize comparison
        visualize_comparison(results)
        
        print("\n=== Test Summary ===")
        print("✓ Original vs improved comparison completed")
        print("✓ Multiple frame testing completed")
        print("✓ Visualization generated")
        
        return results, multi_frame_results
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    results, multi_frame_results = main()
