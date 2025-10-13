"""
Velocity Error Evaluation Module

This module implements evaluation metrics for velocity estimation
following the approach described in "3-D Ego-Motion Estimation Using Multi-Channel FMCW Radar".

Metrics include:
1. RMSE for translational and rotational velocities
2. Bias analysis
3. Error histograms and statistics
4. Comparison with ground truth
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class VelocityErrorEvaluator:
    """
    Evaluates velocity estimation errors against ground truth.
    
    Computes various error metrics and generates evaluation reports.
    """
    
    def __init__(self,
                 velocity_components: List[str] = ['vx', 'vy', 'vz', 'wx', 'wy', 'wz'],
                 error_metrics: List[str] = ['rmse', 'mae', 'bias', 'std']):
        """
        Initialize velocity error evaluator.
        
        Args:
            velocity_components: List of velocity component names
            error_metrics: List of error metrics to compute
        """
        self.velocity_components = velocity_components
        self.error_metrics = error_metrics
        
        logger.info(f"Initialized velocity error evaluator")
        logger.info(f"  Components: {velocity_components}")
        logger.info(f"  Metrics: {error_metrics}")
    
    def compute_velocity_errors(self, 
                               estimated_velocities: np.ndarray,
                               ground_truth_velocities: np.ndarray,
                               timestamps: Optional[np.ndarray] = None) -> Dict:
        """
        Compute velocity estimation errors.
        
        Args:
            estimated_velocities: Estimated velocities [N, 6] (vx, vy, vz, wx, wy, wz)
            ground_truth_velocities: Ground truth velocities [N, 6]
            timestamps: Time stamps [N] (optional)
            
        Returns:
            Error metrics dictionary
        """
        if estimated_velocities.shape != ground_truth_velocities.shape:
            raise ValueError("Estimated and ground truth velocities must have the same shape")
        
        N, num_components = estimated_velocities.shape
        
        if num_components != len(self.velocity_components):
            raise ValueError(f"Expected {len(self.velocity_components)} components, got {num_components}")
        
        # Compute errors
        errors = estimated_velocities - ground_truth_velocities
        
        # Initialize results
        results = {
            'num_samples': N,
            'components': self.velocity_components,
            'errors': errors,
            'estimated_velocities': estimated_velocities,
            'ground_truth_velocities': ground_truth_velocities
        }
        
        if timestamps is not None:
            results['timestamps'] = timestamps
        
        # Compute metrics for each component
        component_metrics = {}
        
        for i, component in enumerate(self.velocity_components):
            component_errors = errors[:, i]
            
            metrics = {}
            
            if 'rmse' in self.error_metrics:
                metrics['rmse'] = np.sqrt(np.mean(component_errors**2))
            
            if 'mae' in self.error_metrics:
                metrics['mae'] = np.mean(np.abs(component_errors))
            
            if 'bias' in self.error_metrics:
                metrics['bias'] = np.mean(component_errors)
            
            if 'std' in self.error_metrics:
                metrics['std'] = np.std(component_errors)
            
            # Additional statistics
            metrics['min_error'] = np.min(component_errors)
            metrics['max_error'] = np.max(component_errors)
            metrics['median_error'] = np.median(component_errors)
            metrics['q25_error'] = np.percentile(component_errors, 25)
            metrics['q75_error'] = np.percentile(component_errors, 75)
            
            component_metrics[component] = metrics
        
        results['component_metrics'] = component_metrics
        
        # Overall metrics
        overall_metrics = {}
        
        if 'rmse' in self.error_metrics:
            overall_metrics['rmse'] = np.sqrt(np.mean(errors**2))
        
        if 'mae' in self.error_metrics:
            overall_metrics['mae'] = np.mean(np.abs(errors))
        
        if 'bias' in self.error_metrics:
            overall_metrics['bias'] = np.mean(errors)
        
        if 'std' in self.error_metrics:
            overall_metrics['std'] = np.std(errors)
        
        results['overall_metrics'] = overall_metrics
        
        logger.info(f"Computed velocity errors for {N} samples")
        logger.info(f"Overall RMSE: {overall_metrics.get('rmse', 0):.6f}")
        logger.info(f"Overall MAE: {overall_metrics.get('mae', 0):.6f}")
        
        return results
    
    def analyze_error_trends(self, 
                           error_results: Dict,
                           window_size: int = 10) -> Dict:
        """
        Analyze error trends over time.
        
        Args:
            error_results: Results from compute_velocity_errors
            window_size: Window size for moving average
            
        Returns:
            Trend analysis dictionary
        """
        errors = error_results['errors']
        timestamps = error_results.get('timestamps', np.arange(len(errors)))
        
        # Moving average of errors
        moving_avg_errors = np.zeros_like(errors)
        for i in range(len(errors)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(errors), i + window_size // 2 + 1)
            moving_avg_errors[i] = np.mean(errors[start_idx:end_idx], axis=0)
        
        # Error drift (linear trend)
        drift_coefficients = np.zeros(errors.shape[1])
        for i in range(errors.shape[1]):
            # Linear regression on errors
            coeffs = np.polyfit(timestamps, errors[:, i], 1)
            drift_coefficients[i] = coeffs[0]  # Slope
        
        # Error stability (variance over time)
        error_variance = np.var(errors, axis=0)
        
        trend_analysis = {
            'moving_avg_errors': moving_avg_errors,
            'drift_coefficients': drift_coefficients,
            'error_variance': error_variance,
            'window_size': window_size
        }
        
        logger.info(f"Analyzed error trends with window size {window_size}")
        
        return trend_analysis
    
    def generate_error_report(self, 
                            error_results: Dict,
                            trend_analysis: Optional[Dict] = None,
                            save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive error report.
        
        Args:
            error_results: Results from compute_velocity_errors
            trend_analysis: Results from analyze_error_trends
            save_path: Path to save report
            
        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("# Velocity Estimation Error Report")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Summary statistics
        overall_metrics = error_results['overall_metrics']
        report_lines.append("## Overall Metrics")
        report_lines.append(f"Number of samples: {error_results['num_samples']}")
        report_lines.append(f"RMSE: {overall_metrics.get('rmse', 0):.6f}")
        report_lines.append(f"MAE: {overall_metrics.get('mae', 0):.6f}")
        report_lines.append(f"Bias: {overall_metrics.get('bias', 0):.6f}")
        report_lines.append(f"Std: {overall_metrics.get('std', 0):.6f}")
        report_lines.append("")
        
        # Component-wise metrics
        report_lines.append("## Component-wise Metrics")
        report_lines.append("")
        
        component_metrics = error_results['component_metrics']
        for component, metrics in component_metrics.items():
            report_lines.append(f"### {component.upper()}")
            report_lines.append(f"RMSE: {metrics['rmse']:.6f}")
            report_lines.append(f"MAE: {metrics['mae']:.6f}")
            report_lines.append(f"Bias: {metrics['bias']:.6f}")
            report_lines.append(f"Std: {metrics['std']:.6f}")
            report_lines.append(f"Min error: {metrics['min_error']:.6f}")
            report_lines.append(f"Max error: {metrics['max_error']:.6f}")
            report_lines.append(f"Median error: {metrics['median_error']:.6f}")
            report_lines.append("")
        
        # Trend analysis
        if trend_analysis is not None:
            report_lines.append("## Error Trend Analysis")
            report_lines.append("")
            
            drift_coeffs = trend_analysis['drift_coefficients']
            error_variance = trend_analysis['error_variance']
            
            for i, component in enumerate(error_results['components']):
                report_lines.append(f"### {component.upper()}")
                report_lines.append(f"Drift coefficient: {drift_coeffs[i]:.6f}")
                report_lines.append(f"Error variance: {error_variance[i]:.6f}")
                report_lines.append("")
        
        # Generate report text
        report_text = "\n".join(report_lines)
        
        # Save report
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Error report saved to {save_path}")
        
        return report_text
    
    def visualize_errors(self, 
                        error_results: Dict,
                        trend_analysis: Optional[Dict] = None,
                        save_path: Optional[str] = None) -> None:
        """
        Visualize velocity estimation errors.
        
        Args:
            error_results: Results from compute_velocity_errors
            trend_analysis: Results from analyze_error_trends
            save_path: Path to save plot
        """
        errors = error_results['errors']
        components = error_results['components']
        timestamps = error_results.get('timestamps', np.arange(len(errors)))
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot errors for each component
        for i, component in enumerate(components):
            ax = axes[i]
            
            # Error over time
            ax.plot(timestamps, errors[:, i], alpha=0.7, linewidth=1)
            
            # Moving average if available
            if trend_analysis is not None:
                moving_avg = trend_analysis['moving_avg_errors'][:, i]
                ax.plot(timestamps, moving_avg, 'r-', linewidth=2, label='Moving Avg')
            
            # Zero line
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Error')
            ax.set_title(f'{component.upper()} Error')
            ax.grid(True, alpha=0.3)
            
            if trend_analysis is not None:
                ax.legend()
        
        # Overall error statistics
        ax = axes[5]
        error_magnitudes = np.linalg.norm(errors, axis=1)
        ax.hist(error_magnitudes, bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Error Magnitude')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Magnitude Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def compare_velocities(self, 
                          error_results: Dict,
                          save_path: Optional[str] = None) -> None:
        """
        Compare estimated vs ground truth velocities.
        
        Args:
            error_results: Results from compute_velocity_errors
            save_path: Path to save plot
        """
        estimated = error_results['estimated_velocities']
        ground_truth = error_results['ground_truth_velocities']
        components = error_results['components']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, component in enumerate(components):
            ax = axes[i]
            
            # Scatter plot
            ax.scatter(ground_truth[:, i], estimated[:, i], alpha=0.6, s=20)
            
            # Perfect correlation line
            min_val = min(ground_truth[:, i].min(), estimated[:, i].min())
            max_val = max(ground_truth[:, i].max(), estimated[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            ax.set_xlabel('Ground Truth')
            ax.set_ylabel('Estimated')
            ax.set_title(f'{component.upper()} Comparison')
            ax.grid(True, alpha=0.3)
            
            # Add correlation coefficient
            correlation = np.corrcoef(ground_truth[:, i], estimated[:, i])[0, 1]
            ax.text(0.05, 0.95, f'R = {correlation:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def evaluate_velocity_errors(estimated_path: str,
                           ground_truth_path: str,
                           output_path: str,
                           timestamps_path: Optional[str] = None) -> Dict:
    """
    Evaluate velocity estimation errors.
    
    Args:
        estimated_path: Path to estimated velocities file
        ground_truth_path: Path to ground truth velocities file
        output_path: Path to save evaluation results
        timestamps_path: Path to timestamps file (optional)
        
    Returns:
        Evaluation results dictionary
    """
    # Load data
    estimated_data = np.load(estimated_path, allow_pickle=True)
    ground_truth_data = np.load(ground_truth_path, allow_pickle=True)
    
    # Extract velocity arrays
    estimated_velocities = np.column_stack([
        estimated_data['velocity'],
        estimated_data['angular_velocity']
    ])
    
    ground_truth_velocities = np.column_stack([
        ground_truth_data['velocity'],
        ground_truth_data['angular_velocity']
    ])
    
    # Load timestamps if available
    timestamps = None
    if timestamps_path and os.path.exists(timestamps_path):
        timestamps = np.load(timestamps_path)
    
    logger.info(f"Loaded velocity data:")
    logger.info(f"  Estimated: {estimated_velocities.shape}")
    logger.info(f"  Ground truth: {ground_truth_velocities.shape}")
    
    # Initialize evaluator
    evaluator = VelocityErrorEvaluator()
    
    # Compute errors
    error_results = evaluator.compute_velocity_errors(
        estimated_velocities, ground_truth_velocities, timestamps
    )
    
    # Analyze trends
    trend_analysis = evaluator.analyze_error_trends(error_results)
    
    # Generate report
    report_path = output_path.replace('.npz', '_report.md')
    report = evaluator.generate_error_report(error_results, trend_analysis, report_path)
    
    # Visualize results
    plot_path = output_path.replace('.npz', '_errors.png')
    evaluator.visualize_errors(error_results, trend_analysis, plot_path)
    
    comparison_path = output_path.replace('.npz', '_comparison.png')
    evaluator.compare_velocities(error_results, comparison_path)
    
    # Save results
    np.savez(output_path, 
             error_results=error_results,
             trend_analysis=trend_analysis,
             report=report)
    
    logger.info(f"Velocity error evaluation complete: {output_path}")
    
    return error_results


if __name__ == "__main__":
    # Example usage
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Evaluate velocity errors')
    parser.add_argument('--est', required=True, help='Path to estimated velocities')
    parser.add_argument('--gt', required=True, help='Path to ground truth velocities')
    parser.add_argument('--out', required=True, help='Output path for evaluation')
    parser.add_argument('--timestamps', help='Path to timestamps file')
    
    args = parser.parse_args()
    
    # Evaluate errors
    results = evaluate_velocity_errors(args.est, args.gt, args.out, args.timestamps)
    print(f"Velocity error evaluation complete: {results['overall_metrics']}")
