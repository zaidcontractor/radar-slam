"""
Pose Error Evaluation Module

This module implements evaluation metrics for pose estimation including
APE (Absolute Pose Error) and RTE (Relative Trajectory Error) as defined
in the paper "3-D Ego-Motion Estimation Using Multi-Channel FMCW Radar".

Metrics include:
1. APE: Absolute pose error at each time step
2. RTE: Relative trajectory error over segments
3. Translation and rotation error analysis
4. Trajectory alignment and comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PoseErrorEvaluator:
    """
    Evaluates pose estimation errors using APE and RTE metrics.
    
    Implements the evaluation framework described in the paper.
    """
    
    def __init__(self,
                 max_ape_threshold: float = 1.0,  # meters
                 max_rte_threshold: float = 0.5,  # meters
                 rte_segment_lengths: List[float] = [100, 200, 300, 400, 500, 600, 700, 800]):  # meters
        """
        Initialize pose error evaluator.
        
        Args:
            max_ape_threshold: Maximum APE threshold for analysis
            max_rte_threshold: Maximum RTE threshold for analysis
            rte_segment_lengths: Segment lengths for RTE computation
        """
        self.max_ape_threshold = max_ape_threshold
        self.max_rte_threshold = max_rte_threshold
        self.rte_segment_lengths = rte_segment_lengths
        
        logger.info(f"Initialized pose error evaluator")
        logger.info(f"  Max APE threshold: {max_ape_threshold} m")
        logger.info(f"  Max RTE threshold: {max_rte_threshold} m")
        logger.info(f"  RTE segment lengths: {rte_segment_lengths} m")
    
    def align_trajectories(self, 
                          estimated_poses: np.ndarray,
                          ground_truth_poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Align estimated trajectory to ground truth using Umeyama algorithm.
        
        Args:
            estimated_poses: Estimated poses [N, 7] (x, y, z, qw, qx, qy, qz)
            ground_truth_poses: Ground truth poses [N, 7] (x, y, z, qw, qx, qy, qz)
            
        Returns:
            Tuple of (aligned_poses, transformation_matrix, alignment_info)
        """
        # Extract positions and orientations
        est_positions = estimated_poses[:, :3]
        gt_positions = ground_truth_poses[:, :3]
        
        est_orientations = estimated_poses[:, 3:7]  # Quaternions
        gt_orientations = ground_truth_poses[:, 3:7]
        
        # Align positions using Umeyama algorithm
        aligned_positions, T_pos = self._umeyama_alignment(est_positions, gt_positions)
        
        # Align orientations
        aligned_orientations, T_rot = self._align_orientations(est_orientations, gt_orientations)
        
        # Combine aligned poses
        aligned_poses = np.column_stack([aligned_positions, aligned_orientations])
        
        # Transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = T_rot
        transformation_matrix[:3, 3] = T_pos[:3, 3]
        
        alignment_info = {
            'position_translation': T_pos[:3, 3],
            'position_rotation': T_pos[:3, :3],
            'orientation_rotation': T_rot,
            'scale_factor': np.linalg.det(T_pos[:3, :3])**(1/3)
        }
        
        logger.info(f"Aligned trajectories:")
        logger.info(f"  Translation: {alignment_info['position_translation']}")
        logger.info(f"  Scale factor: {alignment_info['scale_factor']:.6f}")
        
        return aligned_poses, transformation_matrix, alignment_info
    
    def _umeyama_alignment(self, 
                          source: np.ndarray, 
                          target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Umeyama algorithm for trajectory alignment.
        
        Args:
            source: Source points [N, 3]
            target: Target points [N, 3]
            
        Returns:
            Tuple of (aligned_source, transformation_matrix)
        """
        # Center the points
        source_centered = source - np.mean(source, axis=0)
        target_centered = target - np.mean(target, axis=0)
        
        # Compute cross-covariance matrix
        H = source_centered.T @ target_centered
        
        # SVD
        U, S, Vt = np.linalg.svd(H)
        
        # Compute rotation matrix
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Compute translation
        t = np.mean(target, axis=0) - R @ np.mean(source, axis=0)
        
        # Apply transformation
        aligned_source = (R @ source.T).T + t
        
        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        
        return aligned_source, T
    
    def _align_orientations(self, 
                          source_quats: np.ndarray, 
                          target_quats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align orientations using quaternion alignment.
        
        Args:
            source_quats: Source quaternions [N, 4] (qw, qx, qy, qz)
            target_quats: Target quaternions [N, 4] (qw, qx, qy, qz)
            
        Returns:
            Tuple of (aligned_quaternions, rotation_matrix)
        """
        # Convert to rotation matrices
        source_rots = Rotation.from_quat(source_quats)
        target_rots = Rotation.from_quat(target_quats)
        
        # Compute relative rotations
        relative_rots = target_rots * source_rots.inv()
        
        # Average relative rotation
        avg_relative_rot = relative_rots.mean()
        
        # Apply alignment
        aligned_rots = source_rots * avg_relative_rot
        aligned_quats = aligned_rots.as_quat()
        
        return aligned_quats, avg_relative_rot.as_matrix()
    
    def compute_ape(self, 
                   estimated_poses: np.ndarray,
                   ground_truth_poses: np.ndarray) -> Dict:
        """
        Compute Absolute Pose Error (APE).
        
        Args:
            estimated_poses: Estimated poses [N, 7] (x, y, z, qw, qx, qy, qz)
            ground_truth_poses: Ground truth poses [N, 7] (x, y, z, qw, qx, qy, qz)
            
        Returns:
            APE metrics dictionary
        """
        # Align trajectories
        aligned_poses, T, alignment_info = self.align_trajectories(estimated_poses, ground_truth_poses)
        
        # Extract positions and orientations
        est_positions = aligned_poses[:, :3]
        gt_positions = ground_truth_poses[:, :3]
        
        est_orientations = aligned_poses[:, 3:7]
        gt_orientations = ground_truth_poses[:, 3:7]
        
        # Position errors
        position_errors = np.linalg.norm(est_positions - gt_positions, axis=1)
        
        # Orientation errors
        est_rots = Rotation.from_quat(est_orientations)
        gt_rots = Rotation.from_quat(gt_orientations)
        
        orientation_errors = np.zeros(len(est_rots))
        for i in range(len(est_rots)):
            # Relative rotation
            rel_rot = gt_rots[i] * est_rots[i].inv()
            # Angle of rotation
            orientation_errors[i] = np.linalg.norm(rel_rot.as_rotvec())
        
        # Overall pose errors (combined)
        pose_errors = np.sqrt(position_errors**2 + orientation_errors**2)
        
        # Statistics
        ape_metrics = {
            'position_errors': position_errors,
            'orientation_errors': orientation_errors,
            'pose_errors': pose_errors,
            'position_rmse': np.sqrt(np.mean(position_errors**2)),
            'orientation_rmse': np.sqrt(np.mean(orientation_errors**2)),
            'pose_rmse': np.sqrt(np.mean(pose_errors**2)),
            'position_mean': np.mean(position_errors),
            'orientation_mean': np.mean(orientation_errors),
            'pose_mean': np.mean(pose_errors),
            'position_std': np.std(position_errors),
            'orientation_std': np.std(orientation_errors),
            'pose_std': np.std(pose_errors),
            'position_max': np.max(position_errors),
            'orientation_max': np.max(orientation_errors),
            'pose_max': np.max(pose_errors),
            'alignment_info': alignment_info
        }
        
        logger.info(f"APE computation complete:")
        logger.info(f"  Position RMSE: {ape_metrics['position_rmse']:.6f} m")
        logger.info(f"  Orientation RMSE: {ape_metrics['orientation_rmse']:.6f} rad")
        logger.info(f"  Pose RMSE: {ape_metrics['pose_rmse']:.6f}")
        
        return ape_metrics
    
    def compute_rte(self, 
                   estimated_poses: np.ndarray,
                   ground_truth_poses: np.ndarray,
                   timestamps: Optional[np.ndarray] = None) -> Dict:
        """
        Compute Relative Trajectory Error (RTE).
        
        Args:
            estimated_poses: Estimated poses [N, 7] (x, y, z, qw, qx, qy, qz)
            ground_truth_poses: Ground truth poses [N, 7] (x, y, z, qw, qx, qy, qz)
            timestamps: Time stamps [N] (optional)
            
        Returns:
            RTE metrics dictionary
        """
        # Align trajectories
        aligned_poses, T, alignment_info = self.align_trajectories(estimated_poses, ground_truth_poses)
        
        # Extract positions
        est_positions = aligned_poses[:, :3]
        gt_positions = ground_truth_poses[:, :3]
        
        # Compute trajectory distances
        est_distances = np.cumsum(np.linalg.norm(np.diff(est_positions, axis=0), axis=1))
        est_distances = np.concatenate([[0], est_distances])
        
        gt_distances = np.cumsum(np.linalg.norm(np.diff(gt_positions, axis=0), axis=1))
        gt_distances = np.concatenate([[0], gt_distances])
        
        # RTE for different segment lengths
        rte_metrics = {}
        
        for segment_length in self.rte_segment_lengths:
            # Find segments of specified length
            segment_errors = []
            
            for i in range(len(est_positions)):
                # Find end point for this segment
                end_idx = self._find_segment_end(est_distances, i, segment_length)
                
                if end_idx is not None and end_idx > i:
                    # Compute relative transformation
                    est_rel_trans = self._compute_relative_transformation(
                        est_positions[i], est_positions[end_idx],
                        aligned_poses[i, 3:7], aligned_poses[end_idx, 3:7]
                    )
                    
                    gt_rel_trans = self._compute_relative_transformation(
                        gt_positions[i], gt_positions[end_idx],
                        ground_truth_poses[i, 3:7], ground_truth_poses[end_idx, 3:7]
                    )
                    
                    # Compute relative error
                    rel_error = self._compute_transformation_error(est_rel_trans, gt_rel_trans)
                    segment_errors.append(rel_error)
            
            if segment_errors:
                rte_metrics[f'rte_{segment_length:.0f}m'] = {
                    'errors': np.array(segment_errors),
                    'rmse': np.sqrt(np.mean(np.array(segment_errors)**2)),
                    'mean': np.mean(segment_errors),
                    'std': np.std(segment_errors),
                    'max': np.max(segment_errors),
                    'num_segments': len(segment_errors)
                }
        
        logger.info(f"RTE computation complete for {len(rte_metrics)} segment lengths")
        
        return rte_metrics
    
    def _find_segment_end(self, 
                         distances: np.ndarray, 
                         start_idx: int, 
                         segment_length: float) -> Optional[int]:
        """Find end index for segment of specified length."""
        start_distance = distances[start_idx]
        target_distance = start_distance + segment_length
        
        # Find closest distance
        end_idx = np.searchsorted(distances, target_distance)
        
        if end_idx < len(distances):
            return end_idx
        else:
            return None
    
    def _compute_relative_transformation(self, 
                                       pos1: np.ndarray, 
                                       pos2: np.ndarray,
                                       quat1: np.ndarray, 
                                       quat2: np.ndarray) -> np.ndarray:
        """Compute relative transformation between two poses."""
        # Relative translation
        rel_translation = pos2 - pos1
        
        # Relative rotation
        rot1 = Rotation.from_quat(quat1)
        rot2 = Rotation.from_quat(quat2)
        rel_rotation = rot2 * rot1.inv()
        
        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = rel_rotation.as_matrix()
        T[:3, 3] = rel_translation
        
        return T
    
    def _compute_transformation_error(self, 
                                     T1: np.ndarray, 
                                     T2: np.ndarray) -> float:
        """Compute error between two transformation matrices."""
        # Relative transformation
        T_rel = np.linalg.inv(T1) @ T2
        
        # Translation error
        trans_error = np.linalg.norm(T_rel[:3, 3])
        
        # Rotation error
        rot_error = np.linalg.norm(T_rel[:3, :3] - np.eye(3))
        
        # Combined error
        total_error = np.sqrt(trans_error**2 + rot_error**2)
        
        return total_error
    
    def visualize_ape_rte(self, 
                         ape_metrics: Dict,
                         rte_metrics: Dict,
                         timestamps: Optional[np.ndarray] = None,
                         save_path: Optional[str] = None) -> None:
        """
        Visualize APE and RTE results.
        
        Args:
            ape_metrics: APE metrics dictionary
            rte_metrics: RTE metrics dictionary
            timestamps: Time stamps (optional)
            save_path: Path to save plot
        """
        if timestamps is None:
            timestamps = np.arange(len(ape_metrics['pose_errors']))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # APE over time
        ax = axes[0, 0]
        ax.plot(timestamps, ape_metrics['position_errors'], label='Position', linewidth=2)
        ax.plot(timestamps, ape_metrics['orientation_errors'], label='Orientation', linewidth=2)
        ax.plot(timestamps, ape_metrics['pose_errors'], label='Combined', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error')
        ax.set_title('Absolute Pose Error (APE)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # APE histograms
        ax = axes[0, 1]
        ax.hist(ape_metrics['position_errors'], bins=30, alpha=0.7, label='Position', density=True)
        ax.hist(ape_metrics['orientation_errors'], bins=30, alpha=0.7, label='Orientation', density=True)
        ax.set_xlabel('Error')
        ax.set_ylabel('Density')
        ax.set_title('APE Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # RTE for different segment lengths
        ax = axes[1, 0]
        segment_lengths = []
        rte_means = []
        rte_stds = []
        
        for key, metrics in rte_metrics.items():
            if key.startswith('rte_'):
                segment_length = float(key.split('_')[1].replace('m', ''))
                segment_lengths.append(segment_length)
                rte_means.append(metrics['mean'])
                rte_stds.append(metrics['std'])
        
        if segment_lengths:
            ax.errorbar(segment_lengths, rte_means, yerr=rte_stds, 
                       marker='o', capsize=5, capthick=2)
            ax.set_xlabel('Segment Length (m)')
            ax.set_ylabel('RTE (m)')
            ax.set_title('Relative Trajectory Error (RTE)')
            ax.grid(True, alpha=0.3)
        
        # Statistics summary
        ax = axes[1, 1]
        ax.axis('off')
        
        stats_text = f"APE Statistics:\n"
        stats_text += f"Position RMSE: {ape_metrics['position_rmse']:.4f} m\n"
        stats_text += f"Orientation RMSE: {ape_metrics['orientation_rmse']:.4f} rad\n"
        stats_text += f"Pose RMSE: {ape_metrics['pose_rmse']:.4f}\n\n"
        
        stats_text += f"RTE Statistics:\n"
        for key, metrics in rte_metrics.items():
            if key.startswith('rte_'):
                segment_length = key.split('_')[1].replace('m', '')
                stats_text += f"{segment_length}m: {metrics['rmse']:.4f} m\n"
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def generate_pose_error_report(self, 
                                 ape_metrics: Dict,
                                 rte_metrics: Dict,
                                 save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive pose error report.
        
        Args:
            ape_metrics: APE metrics dictionary
            rte_metrics: RTE metrics dictionary
            save_path: Path to save report
            
        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("# Pose Error Evaluation Report")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # APE Statistics
        report_lines.append("## Absolute Pose Error (APE)")
        report_lines.append("")
        report_lines.append("### Position Errors")
        report_lines.append(f"RMSE: {ape_metrics['position_rmse']:.6f} m")
        report_lines.append(f"Mean: {ape_metrics['position_mean']:.6f} m")
        report_lines.append(f"Std: {ape_metrics['position_std']:.6f} m")
        report_lines.append(f"Max: {ape_metrics['position_max']:.6f} m")
        report_lines.append("")
        
        report_lines.append("### Orientation Errors")
        report_lines.append(f"RMSE: {ape_metrics['orientation_rmse']:.6f} rad")
        report_lines.append(f"Mean: {ape_metrics['orientation_mean']:.6f} rad")
        report_lines.append(f"Std: {ape_metrics['orientation_std']:.6f} rad")
        report_lines.append(f"Max: {ape_metrics['orientation_max']:.6f} rad")
        report_lines.append("")
        
        report_lines.append("### Combined Pose Errors")
        report_lines.append(f"RMSE: {ape_metrics['pose_rmse']:.6f}")
        report_lines.append(f"Mean: {ape_metrics['pose_mean']:.6f}")
        report_lines.append(f"Std: {ape_metrics['pose_std']:.6f}")
        report_lines.append(f"Max: {ape_metrics['pose_max']:.6f}")
        report_lines.append("")
        
        # RTE Statistics
        report_lines.append("## Relative Trajectory Error (RTE)")
        report_lines.append("")
        
        for key, metrics in rte_metrics.items():
            if key.startswith('rte_'):
                segment_length = key.split('_')[1].replace('m', '')
                report_lines.append(f"### {segment_length}m Segments")
                report_lines.append(f"RMSE: {metrics['rmse']:.6f} m")
                report_lines.append(f"Mean: {metrics['mean']:.6f} m")
                report_lines.append(f"Std: {metrics['std']:.6f} m")
                report_lines.append(f"Max: {metrics['max']:.6f} m")
                report_lines.append(f"Number of segments: {metrics['num_segments']}")
                report_lines.append("")
        
        # Generate report text
        report_text = "\n".join(report_lines)
        
        # Save report
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Pose error report saved to {save_path}")
        
        return report_text


def evaluate_pose_errors(estimated_path: str,
                        ground_truth_path: str,
                        output_path: str,
                        timestamps_path: Optional[str] = None) -> Dict:
    """
    Evaluate pose estimation errors using APE and RTE metrics.
    
    Args:
        estimated_path: Path to estimated poses file
        ground_truth_path: Path to ground truth poses file
        output_path: Path to save evaluation results
        timestamps_path: Path to timestamps file (optional)
        
    Returns:
        Evaluation results dictionary
    """
    # Load data
    estimated_data = np.load(estimated_path, allow_pickle=True)
    ground_truth_data = np.load(ground_truth_path, allow_pickle=True)
    
    # Extract pose arrays
    estimated_poses = np.column_stack([
        estimated_data['positions'],
        estimated_data['orientations']
    ])
    
    ground_truth_poses = np.column_stack([
        ground_truth_data['positions'],
        ground_truth_data['orientations']
    ])
    
    # Load timestamps if available
    timestamps = None
    if timestamps_path and os.path.exists(timestamps_path):
        timestamps = np.load(timestamps_path)
    
    logger.info(f"Loaded pose data:")
    logger.info(f"  Estimated: {estimated_poses.shape}")
    logger.info(f"  Ground truth: {ground_truth_poses.shape}")
    
    # Initialize evaluator
    evaluator = PoseErrorEvaluator()
    
    # Compute APE
    ape_metrics = evaluator.compute_ape(estimated_poses, ground_truth_poses)
    
    # Compute RTE
    rte_metrics = evaluator.compute_rte(estimated_poses, ground_truth_poses, timestamps)
    
    # Generate report
    report_path = output_path.replace('.npz', '_report.md')
    report = evaluator.generate_pose_error_report(ape_metrics, rte_metrics, report_path)
    
    # Visualize results
    plot_path = output_path.replace('.npz', '_errors.png')
    evaluator.visualize_ape_rte(ape_metrics, rte_metrics, timestamps, plot_path)
    
    # Save results
    np.savez(output_path, 
             ape_metrics=ape_metrics,
             rte_metrics=rte_metrics,
             report=report)
    
    logger.info(f"Pose error evaluation complete: {output_path}")
    
    return {
        'ape_metrics': ape_metrics,
        'rte_metrics': rte_metrics
    }


if __name__ == "__main__":
    # Example usage
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Evaluate pose errors')
    parser.add_argument('--est', required=True, help='Path to estimated poses')
    parser.add_argument('--gt', required=True, help='Path to ground truth poses')
    parser.add_argument('--out', required=True, help='Output path for evaluation')
    parser.add_argument('--timestamps', help='Path to timestamps file')
    
    args = parser.parse_args()
    
    # Evaluate errors
    results = evaluate_pose_errors(args.est, args.gt, args.out, args.timestamps)
    print(f"Pose error evaluation complete: {results['ape_metrics']['pose_rmse']:.6f}")
