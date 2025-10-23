"""
RadarScenes Dataset Loader

This module provides functionality to load and process RadarScenes dataset:
1. Load radar data from HDF5 files
2. Extract odometry data for ground truth comparison
3. Process radar measurements for ego-motion estimation
4. Provide data visualization and analysis tools
"""

import h5py
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class RadarScenesLoader:
    """
    Loader for RadarScenes dataset with comprehensive data processing.
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize RadarScenes loader.
        
        Args:
            dataset_path: Path to RadarScenes dataset
        """
        self.dataset_path = Path(dataset_path)
        self.sensors_info = self._load_sensors_info()
        self.sequence_info = self._load_sequence_info()
        
        logger.info(f"Initialized RadarScenes loader for: {dataset_path}")
        logger.info(f"Found {len(self.sequence_info)} sequences")
    
    def _load_sensors_info(self) -> Dict:
        """Load sensor configuration information."""
        sensors_file = self.dataset_path / "data" / "sensors.json"
        with open(sensors_file, 'r') as f:
            return json.load(f)
    
    def _load_sequence_info(self) -> Dict:
        """Load sequence information."""
        sequences_file = self.dataset_path / "data" / "sequences.json"
        with open(sequences_file, 'r') as f:
            return json.load(f)
    
    def load_sequence_data(self, sequence_id: str) -> Dict:
        """
        Load complete data for a specific sequence.
        
        Args:
            sequence_id: Sequence identifier (e.g., 'sequence_9')
            
        Returns:
            Dictionary containing radar data, odometry, and metadata
        """
        sequence_path = self.dataset_path / "data" / sequence_id
        
        if not sequence_path.exists():
            raise ValueError(f"Sequence {sequence_id} not found")
        
        # Load radar data
        radar_file = sequence_path / "radar_data.h5"
        with h5py.File(radar_file, 'r') as f:
            radar_data = f['radar_data'][:]
            odometry_data = f['odometry'][:]
        
        # Load scenes metadata
        scenes_file = sequence_path / "scenes.json"
        with open(scenes_file, 'r') as f:
            scenes_data = json.load(f)
        
        # Load camera data info
        camera_path = sequence_path / "camera"
        camera_files = list(camera_path.glob("*.jpg")) if camera_path.exists() else []
        
        # Convert to pandas DataFrames for easier processing
        radar_df = pd.DataFrame(radar_data)
        odometry_df = pd.DataFrame(odometry_data)
        
        # Convert timestamps to datetime
        radar_df['datetime'] = pd.to_datetime(radar_df['timestamp'], unit='us')
        odometry_df['datetime'] = pd.to_datetime(odometry_df['timestamp'], unit='us')
        
        logger.info(f"Loaded sequence {sequence_id}:")
        logger.info(f"  Radar measurements: {len(radar_df)}")
        logger.info(f"  Odometry records: {len(odometry_df)}")
        logger.info(f"  Camera images: {len(camera_files)}")
        logger.info(f"  Duration: {scenes_data['first_timestamp']} to {scenes_data['last_timestamp']}")
        
        return {
            'sequence_id': sequence_id,
            'radar_data': radar_df,
            'odometry_data': odometry_df,
            'scenes_data': scenes_data,
            'camera_files': camera_files,
            'sensors_info': self.sensors_info,
            'metadata': {
                'first_timestamp': scenes_data['first_timestamp'],
                'last_timestamp': scenes_data['last_timestamp'],
                'duration_ms': scenes_data['last_timestamp'] - scenes_data['first_timestamp'],
                'category': scenes_data.get('category', 'unknown')
            }
        }
    
    def get_sequence_statistics(self) -> pd.DataFrame:
        """Get statistics for all sequences."""
        stats_file = self.dataset_path / "sequence_statistics.csv"
        if stats_file.exists():
            return pd.read_csv(stats_file)
        else:
            # Generate basic statistics
            sequences = []
            for seq_dir in (self.dataset_path / "data").iterdir():
                if seq_dir.is_dir() and seq_dir.name.startswith('sequence_'):
                    try:
                        data = self.load_sequence_data(seq_dir.name)
                        sequences.append({
                            'sequence_id': seq_dir.name,
                            'radar_measurements': len(data['radar_data']),
                            'odometry_records': len(data['odometry_data']),
                            'camera_images': len(data['camera_files']),
                            'duration_ms': data['metadata']['duration_ms'],
                            'category': data['metadata']['category']
                        })
                    except Exception as e:
                        logger.warning(f"Could not load {seq_dir.name}: {e}")
            
            return pd.DataFrame(sequences)
    
    def extract_radar_frames(self, 
                            sequence_data: Dict,
                            frame_duration_ms: float = 100.0) -> List[Dict]:
        """
        Extract radar frames for processing.
        
        Args:
            sequence_data: Loaded sequence data
            frame_duration_ms: Frame duration in milliseconds
            
        Returns:
            List of radar frames
        """
        radar_df = sequence_data['radar_data']
        frames = []
        
        # Group radar data by time windows
        start_time = radar_df['timestamp'].min()
        end_time = radar_df['timestamp'].max()
        
        current_time = start_time
        frame_id = 0
        
        while current_time < end_time:
            frame_end_time = current_time + frame_duration_ms * 1000  # Convert to microseconds
            
            # Get radar measurements in this time window
            frame_data = radar_df[
                (radar_df['timestamp'] >= current_time) & 
                (radar_df['timestamp'] < frame_end_time)
            ].copy()
            
            if len(frame_data) > 0:
                # Group by sensor
                sensor_groups = {}
                for sensor_id in frame_data['sensor_id'].unique():
                    sensor_data = frame_data[frame_data['sensor_id'] == sensor_id]
                    sensor_groups[sensor_id] = sensor_data
                
                frame = {
                    'frame_id': frame_id,
                    'timestamp': current_time,
                    'frame_end_time': frame_end_time,
                    'sensor_data': sensor_groups,
                    'total_measurements': len(frame_data),
                    'sensors': list(sensor_groups.keys())
                }
                frames.append(frame)
                frame_id += 1
            
            current_time = frame_end_time
        
        logger.info(f"Extracted {len(frames)} radar frames")
        return frames
    
    def get_odometry_at_time(self, 
                            sequence_data: Dict,
                            timestamp: int) -> Optional[Dict]:
        """
        Get odometry data at specific timestamp.
        
        Args:
            sequence_data: Loaded sequence data
            timestamp: Target timestamp in microseconds
            
        Returns:
            Odometry data or None if not found
        """
        odometry_df = sequence_data['odometry_data']
        
        # Find closest odometry record
        time_diffs = np.abs(odometry_df['timestamp'] - timestamp)
        closest_idx = time_diffs.idxmin()
        
        if time_diffs.iloc[closest_idx] < 1e6:  # Within 1 second
            odometry_record = odometry_df.iloc[closest_idx]
            return {
                'timestamp': odometry_record['timestamp'],
                'x': odometry_record['x_seq'],
                'y': odometry_record['y_seq'],
                'yaw': odometry_record['yaw_seq'],
                'vx': odometry_record['vx'],
                'yaw_rate': odometry_record['yaw_rate']
            }
        
        return None
    
    def convert_radar_to_scatterers(self, 
                                  frame_data: Dict,
                                  sensor_id: int) -> pd.DataFrame:
        """
        Convert radar frame data to scatterer format for processing.
        
        Args:
            frame_data: Radar frame data
            sensor_id: Sensor ID to process
            
        Returns:
            DataFrame in scatterer format
        """
        if sensor_id not in frame_data['sensor_data']:
            return pd.DataFrame()
        
        sensor_data = frame_data['sensor_data'][sensor_id]
        
        # Convert to scatterer format
        scatterers = pd.DataFrame({
            'range_sc': sensor_data['range_sc'].values,
            'azimuth_sc': sensor_data['azimuth_sc'].values,
            'rcs': sensor_data['rcs'].values,
            'vr': sensor_data['vr'].values,
            'x_cc': sensor_data['x_cc'].values,
            'y_cc': sensor_data['y_cc'].values
        })
        
        return scatterers
    
    def visualize_sequence_overview(self, 
                                  sequence_data: Dict,
                                  save_path: Optional[str] = None) -> None:
        """
        Create comprehensive visualization of sequence data.
        
        Args:
            sequence_data: Loaded sequence data
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        radar_df = sequence_data['radar_data']
        odometry_df = sequence_data['odometry_data']
        
        # 1. Radar measurements over time
        ax = axes[0, 0]
        radar_df.plot(x='datetime', y='range_sc', kind='scatter', 
                     alpha=0.5, s=1, ax=ax, c='blue')
        ax.set_title('Radar Range Measurements Over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Range (m)')
        
        # 2. Azimuth distribution
        ax = axes[0, 1]
        ax.hist(radar_df['azimuth_sc'], bins=50, alpha=0.7, edgecolor='black')
        ax.set_title('Azimuth Distribution')
        ax.set_xlabel('Azimuth (rad)')
        ax.set_ylabel('Count')
        
        # 3. Range distribution
        ax = axes[0, 2]
        ax.hist(radar_df['range_sc'], bins=50, alpha=0.7, edgecolor='black', color='green')
        ax.set_title('Range Distribution')
        ax.set_xlabel('Range (m)')
        ax.set_ylabel('Count')
        
        # 4. Odometry trajectory
        ax = axes[1, 0]
        ax.plot(odometry_df['x_seq'], odometry_df['y_seq'], 'b-', linewidth=2)
        ax.set_title('Vehicle Trajectory (Odometry)')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # 5. Velocity profile
        ax = axes[1, 1]
        ax.plot(odometry_df['datetime'], odometry_df['vx'], 'r-', linewidth=2)
        ax.set_title('Velocity Profile')
        ax.set_xlabel('Time')
        ax.set_ylabel('Velocity (m/s)')
        ax.grid(True, alpha=0.3)
        
        # 6. Yaw rate
        ax = axes[1, 2]
        ax.plot(odometry_df['datetime'], odometry_df['yaw_rate'], 'g-', linewidth=2)
        ax.set_title('Yaw Rate Profile')
        ax.set_xlabel('Time')
        ax.set_ylabel('Yaw Rate (rad/s)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def analyze_sequence_quality(self, sequence_data: Dict) -> Dict:
        """
        Analyze sequence data quality and characteristics.
        
        Args:
            sequence_data: Loaded sequence data
            
        Returns:
            Quality analysis results
        """
        radar_df = sequence_data['radar_data']
        odometry_df = sequence_data['odometry_data']
        
        # Basic statistics
        analysis = {
            'sequence_id': sequence_data['sequence_id'],
            'duration_seconds': sequence_data['metadata']['duration_ms'] / 1000.0,
            'radar_measurements': len(radar_df),
            'odometry_records': len(odometry_df),
            'camera_images': len(sequence_data['camera_files']),
            'sensors_used': radar_df['sensor_id'].unique().tolist(),
            'measurement_rate': len(radar_df) / (sequence_data['metadata']['duration_ms'] / 1000.0),
            'odometry_rate': len(odometry_df) / (sequence_data['metadata']['duration_ms'] / 1000.0)
        }
        
        # Range and azimuth statistics
        analysis.update({
            'range_stats': {
                'min': radar_df['range_sc'].min(),
                'max': radar_df['range_sc'].max(),
                'mean': radar_df['range_sc'].mean(),
                'std': radar_df['range_sc'].std()
            },
            'azimuth_stats': {
                'min': radar_df['azimuth_sc'].min(),
                'max': radar_df['azimuth_sc'].max(),
                'mean': radar_df['azimuth_sc'].mean(),
                'std': radar_df['azimuth_sc'].std()
            }
        })
        
        # Velocity statistics
        if len(odometry_df) > 0:
            analysis.update({
                'velocity_stats': {
                    'min': odometry_df['vx'].min(),
                    'max': odometry_df['vx'].max(),
                    'mean': odometry_df['vx'].mean(),
                    'std': odometry_df['vx'].std()
                },
                'yaw_rate_stats': {
                    'min': odometry_df['yaw_rate'].min(),
                    'max': odometry_df['yaw_rate'].max(),
                    'mean': odometry_df['yaw_rate'].mean(),
                    'std': odometry_df['yaw_rate'].std()
                }
            })
        
        # Trajectory analysis
        if len(odometry_df) > 1:
            x_diff = odometry_df['x_seq'].diff()
            y_diff = odometry_df['y_seq'].diff()
            distances = np.sqrt(x_diff**2 + y_diff**2)
            total_distance = distances.sum()
            
            analysis.update({
                'total_distance': total_distance,
                'average_speed': total_distance / analysis['duration_seconds'] if analysis['duration_seconds'] > 0 else 0
            })
        
        return analysis


def load_radarscenes_sequence(dataset_path: str, sequence_id: str) -> Dict:
    """
    Convenience function to load a specific RadarScenes sequence.
    
    Args:
        dataset_path: Path to RadarScenes dataset
        sequence_id: Sequence identifier
        
    Returns:
        Loaded sequence data
    """
    loader = RadarScenesLoader(dataset_path)
    return loader.load_sequence_data(sequence_id)


def analyze_radarscenes_dataset(dataset_path: str) -> Dict:
    """
    Analyze entire RadarScenes dataset.
    
    Args:
        dataset_path: Path to RadarScenes dataset
        
    Returns:
        Dataset analysis results
    """
    loader = RadarScenesLoader(dataset_path)
    
    # Get sequence statistics
    sequence_stats = loader.get_sequence_statistics()
    
    # Analyze a few representative sequences
    sample_sequences = ['sequence_9', 'sequence_10', 'sequence_11']
    detailed_analyses = []
    
    for seq_id in sample_sequences:
        try:
            sequence_data = loader.load_sequence_data(seq_id)
            analysis = loader.analyze_sequence_quality(sequence_data)
            detailed_analyses.append(analysis)
        except Exception as e:
            logger.warning(f"Could not analyze {seq_id}: {e}")
    
    return {
        'sequence_statistics': sequence_stats,
        'detailed_analyses': detailed_analyses,
        'total_sequences': len(sequence_stats),
        'dataset_path': dataset_path
    }


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Load and analyze RadarScenes dataset')
    parser.add_argument('--dataset', required=True, help='Path to RadarScenes dataset')
    parser.add_argument('--sequence', help='Specific sequence to analyze')
    parser.add_argument('--analyze-all', action='store_true', help='Analyze entire dataset')
    
    args = parser.parse_args()
    
    if args.analyze_all:
        # Analyze entire dataset
        results = analyze_radarscenes_dataset(args.dataset)
        print(f"Dataset analysis complete: {results}")
    elif args.sequence:
        # Analyze specific sequence
        sequence_data = load_radarscenes_sequence(args.dataset, args.sequence)
        analysis = RadarScenesLoader(args.dataset).analyze_sequence_quality(sequence_data)
        print(f"Sequence analysis: {analysis}")
    else:
        print("Please specify --sequence or --analyze-all")
