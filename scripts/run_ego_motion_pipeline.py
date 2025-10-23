#!/usr/bin/env python3
"""
Complete Ego-Motion Estimation Pipeline

This script runs the complete pipeline for 3D ego-motion estimation
following the paper "3-D Ego-Motion Estimation Using Multi-Channel FMCW Radar".

Pipeline steps:
1. Synthesize raw FMCW signals from RadarScenes point clouds
2. Process signals (dechirp, windowing, 2D FFT)
3. Extract angles of arrival per target
4. Estimate velocity using two-step optimization
5. Integrate velocities to pose trajectory
6. Evaluate against ground truth
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'evaluation'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from simulate_raw import FMCWRadarSimulator
from src.radar_signal.dechirp import SignalPreprocessor
from src.angle_estimation.angle_estimation import AngleEstimator
from src.velocity_solver.velocity_solver import VelocitySolver
from src.pose_integration.pose_integration import PoseIntegrator
from evaluation.compute_velocity_error import VelocityErrorEvaluator
from evaluation.compute_pose_error import PoseErrorEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ego_motion_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EgoMotionPipeline:
    """
    Complete ego-motion estimation pipeline.
    
    Implements the full pipeline described in the paper.
    """
    
    def __init__(self,
                 sequence_name: str,
                 radar_scenes_path: str,
                 output_dir: str,
                 radar_params: Optional[Dict] = None,
                 max_frames: Optional[int] = None):
        """
        Initialize pipeline.
        
        Args:
            sequence_name: RadarScenes sequence name
            radar_scenes_path: Path to RadarScenes dataset
            output_dir: Output directory for results
            radar_params: Radar parameters dictionary
            max_frames: Maximum number of frames to process
        """
        self.sequence_name = sequence_name
        self.radar_scenes_path = radar_scenes_path
        self.output_dir = Path(output_dir)
        self.max_frames = max_frames
        
        # Default radar parameters
        if radar_params is None:
            radar_params = {
                'fc': 77e9,
                'bandwidth': 1e9,
                'chirp_duration': 40e-6,
                'pri': 100e-6,
                'num_chirps': 64,
                'num_antennas': 8,
                'antenna_spacing': 3e8 / (2 * 77e9),
                'sampling_rate': 10e6,
                'noise_power': 0.01
            }
        
        self.radar_params = radar_params
        
        # Create output directories
        self.raw_dir = self.output_dir / 'raw_sim'
        self.rds_dir = self.output_dir / 'rds'
        self.angles_dir = self.output_dir / 'angles'
        self.velocities_dir = self.output_dir / 'velocities'
        self.poses_dir = self.output_dir / 'poses'
        self.evaluation_dir = self.output_dir / 'evaluation'
        
        for dir_path in [self.raw_dir, self.rds_dir, self.angles_dir, 
                        self.velocities_dir, self.poses_dir, self.evaluation_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized pipeline for sequence: {sequence_name}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Max frames: {max_frames}")
    
    def step1_synthesize_raw_signals(self) -> Dict:
        """
        Step 1: Synthesize raw FMCW signals from RadarScenes point clouds.
        
        Returns:
            Synthesis statistics
        """
        logger.info("Step 1: Synthesizing raw FMCW signals...")
        
        # Initialize simulator
        simulator = FMCWRadarSimulator(**self.radar_params)
        
        # Process sequence
        sequence_path = os.path.join(self.radar_scenes_path, self.sequence_name)
        stats = simulator.process_sequence(
            sequence_path=sequence_path,
            output_path=str(self.raw_dir),
            max_frames=self.max_frames
        )
        
        logger.info(f"Raw signal synthesis complete: {stats}")
        return stats
    
    def step2_process_signals(self) -> Dict:
        """
        Step 2: Process raw signals (dechirp, windowing, 2D FFT).
        
        Returns:
            Processing statistics
        """
        logger.info("Step 2: Processing raw signals...")
        
        # Initialize preprocessor (remove num_antennas and antenna_spacing from params)
        preprocessor_params = {k: v for k, v in self.radar_params.items() 
                             if k not in ['num_antennas', 'antenna_spacing', 'noise_power']}
        preprocessor = SignalPreprocessor(**preprocessor_params)
        
        # Process all frames
        frame_files = list(self.raw_dir.glob('frame_*.npy'))
        processed_frames = 0
        
        for frame_file in frame_files:
            try:
                # Load raw signals
                raw_signals = np.load(frame_file)
                
                # Generate RDS
                rds = preprocessor.generate_range_doppler_spectrum(raw_signals)
                
                # Extract peaks
                peak_info = preprocessor.extract_range_doppler_peaks(rds)
                
                # Save RDS
                rds_file = self.rds_dir / f"{frame_file.stem}_rds.npy"
                np.save(rds_file, rds)
                
                # Save peak info
                peak_file = self.rds_dir / f"{frame_file.stem}_peaks.npz"
                np.savez(peak_file, **peak_info)
                
                processed_frames += 1
                
                if processed_frames % 10 == 0:
                    logger.info(f"Processed {processed_frames} frames...")
                
            except Exception as e:
                logger.warning(f"Error processing frame {frame_file}: {e}")
                continue
        
        logger.info(f"Signal processing complete: {processed_frames} frames")
        return {'processed_frames': processed_frames}
    
    def step3_extract_angles(self) -> Dict:
        """
        Step 3: Extract angles of arrival per target.
        
        Returns:
            Angle extraction statistics
        """
        logger.info("Step 3: Extracting angles of arrival...")
        
        # Initialize angle estimator
        estimator = AngleEstimator(
            fc=self.radar_params['fc'],
            antenna_spacing=self.radar_params['antenna_spacing'],
            num_antennas=self.radar_params['num_antennas']
        )
        
        # Process all frames
        rds_files = list(self.rds_dir.glob('*_rds.npy'))
        processed_frames = 0
        total_targets = 0
        
        for rds_file in rds_files:
            try:
                # Load RDS and peak info
                rds = np.load(rds_file)
                # Fix file path construction - remove '_rds' from stem
                base_name = rds_file.stem.replace('_rds', '')
                peak_file = rds_file.parent / f"{base_name}_peaks.npz"
                peak_data = np.load(peak_file, allow_pickle=True)
                peak_info = dict(peak_data)
                
                # Extract angles
                targets = estimator.process_targets(rds, peak_info, method='music')
                
                # Save angle estimates
                angles_file = self.angles_dir / f"{rds_file.stem}_angles.npz"
                np.savez(angles_file, targets=targets, radar_params=self.radar_params)
                
                processed_frames += 1
                total_targets += len(targets)
                
                if processed_frames % 10 == 0:
                    logger.info(f"Processed {processed_frames} frames...")
                
            except Exception as e:
                logger.warning(f"Error processing frame {rds_file}: {e}")
                continue
        
        logger.info(f"Angle extraction complete: {processed_frames} frames, {total_targets} targets")
        return {'processed_frames': processed_frames, 'total_targets': total_targets}
    
    def step4_estimate_velocity(self) -> Dict:
        """
        Step 4: Estimate velocity using two-step optimization.
        
        Returns:
            Velocity estimation statistics
        """
        logger.info("Step 4: Estimating velocity...")
        
        # Initialize velocity solver
        solver = VelocitySolver(
            fc=self.radar_params['fc'],
            lambda_c=self.radar_params['fc'] / 3e8,
            num_antennas=self.radar_params['num_antennas'],
            antenna_spacing=self.radar_params['antenna_spacing']
        )
        
        # Process all frames
        angles_files = list(self.angles_dir.glob('*_angles.npz'))
        processed_frames = 0
        successful_estimates = 0
        
        for angles_file in angles_files:
            try:
                # Load angle estimates
                angles_data = np.load(angles_file, allow_pickle=True)
                targets = angles_data['targets']
                
                # Find corresponding RDS file
                rds_file = self.rds_dir / f"{angles_file.stem.replace('_angles', '_rds')}.npy"
                if not rds_file.exists():
                    continue
                
                rds = np.load(rds_file)
                
                # Estimate velocity
                results = solver.solve_velocity(rds, targets, dt=0.1)
                
                if results['success']:
                    # Save velocity estimates
                    velocity_file = self.velocities_dir / f"{angles_file.stem}_velocity.npz"
                    np.savez(velocity_file, **results)
                    
                    successful_estimates += 1
                
                processed_frames += 1
                
                if processed_frames % 10 == 0:
                    logger.info(f"Processed {processed_frames} frames...")
                
            except Exception as e:
                logger.warning(f"Error processing frame {angles_file}: {e}")
                continue
        
        logger.info(f"Velocity estimation complete: {processed_frames} frames, {successful_estimates} successful")
        return {'processed_frames': processed_frames, 'successful_estimates': successful_estimates}
    
    def step5_integrate_pose(self) -> Dict:
        """
        Step 5: Integrate velocities to pose trajectory.
        
        Returns:
            Pose integration statistics
        """
        logger.info("Step 5: Integrating pose trajectory...")
        
        # Initialize pose integrator
        integrator = PoseIntegrator()
        
        # Collect all velocity estimates
        velocity_files = list(self.velocities_dir.glob('*_velocity.npz'))
        
        if not velocity_files:
            logger.warning("No velocity files found for pose integration")
            return {'success': False}
        
        # Load velocity data
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
                logger.warning(f"Error loading velocity file {velocity_file}: {e}")
                continue
        
        if not velocities:
            logger.warning("No valid velocity estimates found")
            return {'success': False}
        
        # Convert to arrays
        velocities = np.array(velocities)
        angular_velocities = np.array(angular_velocities)
        timestamps = np.array(timestamps)
        
        # Integrate to pose
        trajectory = integrator.integrate_pose(velocities, angular_velocities, timestamps)
        
        # Save trajectory
        trajectory_file = self.poses_dir / 'trajectory.npz'
        integrator.save_trajectory(trajectory, str(trajectory_file))
        
        logger.info(f"Pose integration complete: {len(trajectory['positions'])} points")
        return {'success': True, 'trajectory_points': len(trajectory['positions'])}
    
    def step6_evaluate_results(self, ground_truth_path: Optional[str] = None) -> Dict:
        """
        Step 6: Evaluate results against ground truth.
        
        Args:
            ground_truth_path: Path to ground truth data (optional)
            
        Returns:
            Evaluation statistics
        """
        logger.info("Step 6: Evaluating results...")
        
        # Check if ground truth is available
        if ground_truth_path is None:
            logger.info("No ground truth provided, skipping evaluation")
            return {'evaluation': 'skipped'}
        
        # Load ground truth data
        try:
            gt_data = np.load(ground_truth_path, allow_pickle=True)
            logger.info(f"Loaded ground truth data: {gt_data.files}")
        except Exception as e:
            logger.warning(f"Error loading ground truth: {e}")
            return {'evaluation': 'failed'}
        
        # Evaluate velocity errors
        velocity_files = list(self.velocities_dir.glob('*_velocity.npz'))
        if velocity_files:
            # Collect velocity estimates
            estimated_velocities = []
            estimated_angular_velocities = []
            
            for velocity_file in sorted(velocity_files):
                try:
                    velocity_data = np.load(velocity_file, allow_pickle=True)
                    if velocity_data['success']:
                        estimated_velocities.append(velocity_data['velocity'])
                        estimated_angular_velocities.append(velocity_data['angular_velocity'])
                except:
                    continue
            
            if estimated_velocities:
                # Save velocity estimates for evaluation
                velocity_est_file = self.evaluation_dir / 'estimated_velocities.npz'
                np.savez(velocity_est_file, 
                        velocity=np.array(estimated_velocities),
                        angular_velocity=np.array(estimated_angular_velocities))
                
                logger.info(f"Saved velocity estimates for evaluation: {len(estimated_velocities)} points")
        
        # Evaluate pose errors
        trajectory_file = self.poses_dir / 'trajectory.npz'
        if trajectory_file.exists():
            logger.info("Pose trajectory available for evaluation")
        
        logger.info("Evaluation setup complete")
        return {'evaluation': 'setup_complete'}
    
    def run_complete_pipeline(self, ground_truth_path: Optional[str] = None) -> Dict:
        """
        Run the complete ego-motion estimation pipeline.
        
        Args:
            ground_truth_path: Path to ground truth data (optional)
            
        Returns:
            Pipeline results dictionary
        """
        logger.info("Starting complete ego-motion estimation pipeline...")
        
        results = {}
        
        try:
            # Step 1: Synthesize raw signals
            results['step1'] = self.step1_synthesize_raw_signals()
            
            # Step 2: Process signals
            results['step2'] = self.step2_process_signals()
            
            # Step 3: Extract angles
            results['step3'] = self.step3_extract_angles()
            
            # Step 4: Estimate velocity
            results['step4'] = self.step4_estimate_velocity()
            
            # Step 5: Integrate pose
            results['step5'] = self.step5_integrate_pose()
            
            # Step 6: Evaluate results
            results['step6'] = self.step6_evaluate_results(ground_truth_path)
            
            logger.info("Pipeline execution complete!")
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            results['error'] = str(e)
        
        return results


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Run ego-motion estimation pipeline')
    parser.add_argument('--seq', required=True, help='RadarScenes sequence name')
    parser.add_argument('--radar-scenes', required=True, help='Path to RadarScenes dataset')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--max-frames', type=int, help='Maximum number of frames to process')
    parser.add_argument('--ground-truth', help='Path to ground truth data')
    parser.add_argument('--fc', type=float, default=77e9, help='Carrier frequency (Hz)')
    parser.add_argument('--bandwidth', type=float, default=1e9, help='Bandwidth (Hz)')
    parser.add_argument('--chirp-duration', type=float, default=40e-6, help='Chirp duration (s)')
    parser.add_argument('--pri', type=float, default=100e-6, help='Pulse repetition interval (s)')
    parser.add_argument('--num-chirps', type=int, default=64, help='Number of chirps per frame')
    parser.add_argument('--num-antennas', type=int, default=8, help='Number of antenna elements')
    parser.add_argument('--sampling-rate', type=float, default=10e6, help='Sampling rate (Hz)')
    parser.add_argument('--noise-power', type=float, default=0.01, help='Noise power')
    
    args = parser.parse_args()
    
    # Radar parameters
    radar_params = {
        'fc': args.fc,
        'bandwidth': args.bandwidth,
        'chirp_duration': args.chirp_duration,
        'pri': args.pri,
        'num_chirps': args.num_chirps,
        'num_antennas': args.num_antennas,
        'antenna_spacing': 3e8 / (2 * args.fc),
        'sampling_rate': args.sampling_rate,
        'noise_power': args.noise_power
    }
    
    # Initialize pipeline
    pipeline = EgoMotionPipeline(
        sequence_name=args.seq,
        radar_scenes_path=args.radar_scenes,
        output_dir=args.output,
        radar_params=radar_params,
        max_frames=args.max_frames
    )
    
    # Run pipeline
    results = pipeline.run_complete_pipeline(args.ground_truth)
    
    # Print results
    print("\nPipeline Results:")
    print("=" * 50)
    for step, result in results.items():
        print(f"{step}: {result}")
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
