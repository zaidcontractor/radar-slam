"""
Real-Time Processing Module

This module implements real-time processing optimizations:
1. Frame buffering for temporal processing
2. Parallel processing for multi-target scenarios
3. Computational efficiency optimizations
4. Memory management for continuous operation
"""

import numpy as np
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Tuple, Optional, Callable
import logging
import queue
import multiprocessing as mp
from dataclasses import dataclass
try:
    import psutil
except ImportError:
    psutil = None
import gc

logger = logging.getLogger(__name__)

@dataclass
class ProcessingFrame:
    """Data structure for processing frames."""
    frame_id: int
    timestamp: float
    rds_data: np.ndarray
    peak_info: Dict
    targets: List[Dict]
    processing_time: float
    memory_usage: float

class FrameBuffer:
    """
    Frame buffer for temporal processing with memory management.
    """
    
    def __init__(self, 
                 max_frames: int = 10,
                 max_memory_mb: float = 1000.0):
        """
        Initialize frame buffer.
        
        Args:
            max_frames: Maximum number of frames to buffer
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_frames = max_frames
        self.max_memory_mb = max_memory_mb
        self.frames = deque(maxlen=max_frames)
        self.lock = threading.Lock()
        
        logger.info(f"Initialized frame buffer:")
        logger.info(f"  Max frames: {max_frames}")
        logger.info(f"  Max memory: {max_memory_mb} MB")
    
    def add_frame(self, frame: ProcessingFrame) -> None:
        """Add frame to buffer."""
        with self.lock:
            self.frames.append(frame)
            
            # Check memory usage
            current_memory = self.get_memory_usage()
            if current_memory > self.max_memory_mb:
                logger.warning(f"Memory usage {current_memory:.1f} MB exceeds limit {self.max_memory_mb} MB")
                self.cleanup_old_frames()
    
    def get_frame(self, frame_id: int) -> Optional[ProcessingFrame]:
        """Get frame by ID."""
        with self.lock:
            for frame in self.frames:
                if frame.frame_id == frame_id:
                    return frame
            return None
    
    def get_latest_frames(self, n: int = 2) -> List[ProcessingFrame]:
        """Get latest N frames."""
        with self.lock:
            return list(self.frames)[-n:] if len(self.frames) >= n else list(self.frames)
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        total_memory = 0.0
        for frame in self.frames:
            # Estimate memory usage
            rds_memory = frame.rds_data.nbytes / (1024 * 1024)
            targets_memory = len(frame.targets) * 0.001  # Rough estimate
            total_memory += rds_memory + targets_memory + frame.memory_usage
        return total_memory
    
    def cleanup_old_frames(self) -> None:
        """Cleanup old frames to reduce memory usage."""
        with self.lock:
            if len(self.frames) > self.max_frames // 2:
                # Remove oldest frames
                frames_to_remove = len(self.frames) - self.max_frames // 2
                for _ in range(frames_to_remove):
                    self.frames.popleft()
                
                # Force garbage collection
                gc.collect()
                logger.info(f"Cleaned up {frames_to_remove} old frames")

class ParallelTargetProcessor:
    """
    Parallel processor for multi-target scenarios.
    """
    
    def __init__(self, 
                 max_workers: int = None,
                 use_processes: bool = False):
        """
        Initialize parallel target processor.
        
        Args:
            max_workers: Maximum number of worker threads/processes
            use_processes: Whether to use processes instead of threads
        """
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.use_processes = use_processes
        self.executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        logger.info(f"Initialized parallel target processor:")
        logger.info(f"  Max workers: {self.max_workers}")
        logger.info(f"  Use processes: {use_processes}")
    
    def process_targets_parallel(self, 
                                targets: List[Dict],
                                processing_function: Callable,
                                chunk_size: int = 10) -> List[Dict]:
        """
        Process targets in parallel.
        
        Args:
            targets: List of targets to process
            processing_function: Function to process each target
            chunk_size: Size of chunks for parallel processing
            
        Returns:
            List of processed targets
        """
        if len(targets) <= chunk_size:
            # Process sequentially for small batches
            return [processing_function(target) for target in targets]
        
        # Split targets into chunks
        target_chunks = [targets[i:i + chunk_size] for i in range(0, len(targets), chunk_size)]
        
        # Process chunks in parallel
        with self.executor_class(max_workers=self.max_workers) as executor:
            futures = []
            for chunk in target_chunks:
                future = executor.submit(self._process_chunk, chunk, processing_function)
                futures.append(future)
            
            # Collect results
            results = []
            for future in futures:
                chunk_results = future.result()
                results.extend(chunk_results)
        
        return results
    
    def _process_chunk(self, 
                      target_chunk: List[Dict],
                      processing_function: Callable) -> List[Dict]:
        """Process a chunk of targets."""
        return [processing_function(target) for target in target_chunk]

class RealTimeProcessor:
    """
    Real-time processor with optimizations for continuous operation.
    """
    
    def __init__(self,
                 frame_buffer_size: int = 10,
                 max_memory_mb: float = 1000.0,
                 use_parallel: bool = True,
                 max_workers: int = None,
                 target_chunk_size: int = 10):
        """
        Initialize real-time processor.
        
        Args:
            frame_buffer_size: Size of frame buffer
            max_memory_mb: Maximum memory usage
            use_parallel: Whether to use parallel processing
            max_workers: Maximum number of workers
            target_chunk_size: Size of target chunks for parallel processing
        """
        self.frame_buffer = FrameBuffer(frame_buffer_size, max_memory_mb)
        self.parallel_processor = ParallelTargetProcessor(max_workers) if use_parallel else None
        self.target_chunk_size = target_chunk_size
        self.use_parallel = use_parallel
        
        # Performance monitoring
        self.processing_times = deque(maxlen=100)
        self.memory_usage_history = deque(maxlen=100)
        self.frame_count = 0
        
        # Processing queue for asynchronous processing
        self.processing_queue = queue.Queue(maxsize=5)
        self.processing_thread = None
        self.is_processing = False
        
        logger.info(f"Initialized real-time processor:")
        logger.info(f"  Frame buffer size: {frame_buffer_size}")
        logger.info(f"  Max memory: {max_memory_mb} MB")
        logger.info(f"  Parallel processing: {use_parallel}")
        logger.info(f"  Target chunk size: {target_chunk_size}")
    
    def start_processing(self) -> None:
        """Start background processing thread."""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.is_processing = True
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.start()
            logger.info("Started background processing thread")
    
    def stop_processing(self) -> None:
        """Stop background processing thread."""
        self.is_processing = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        logger.info("Stopped background processing thread")
    
    def _processing_loop(self) -> None:
        """Background processing loop."""
        while self.is_processing:
            try:
                # Get frame from queue with timeout
                frame_data = self.processing_queue.get(timeout=1.0)
                self._process_frame_async(frame_data)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
    
    def add_frame_for_processing(self, 
                                rds_data: np.ndarray,
                                peak_info: Dict,
                                frame_id: int = None,
                                timestamp: float = None) -> int:
        """
        Add frame for processing.
        
        Args:
            rds_data: RDS data
            peak_info: Peak information
            frame_id: Frame ID (auto-generated if None)
            timestamp: Timestamp (current time if None)
            
        Returns:
            Frame ID
        """
        if frame_id is None:
            frame_id = self.frame_count
            self.frame_count += 1
        
        if timestamp is None:
            timestamp = time.time()
        
        # Create frame data
        frame_data = {
            'frame_id': frame_id,
            'timestamp': timestamp,
            'rds_data': rds_data,
            'peak_info': peak_info
        }
        
        # Add to processing queue
        try:
            self.processing_queue.put(frame_data, timeout=0.1)
        except queue.Full:
            logger.warning("Processing queue full, dropping frame")
        
        return frame_id
    
    def _process_frame_async(self, frame_data: Dict) -> None:
        """Process frame asynchronously."""
        start_time = time.time()
        
        try:
            # Get memory usage
            if psutil:
                memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            else:
                memory_usage = 0.0  # Fallback when psutil not available
            
            # Process frame (this would call your actual processing functions)
            # For now, we'll simulate processing
            rds_data = frame_data['rds_data']
            peak_info = frame_data['peak_info']
            
            # Simulate target processing
            targets = self._simulate_target_processing(rds_data, peak_info)
            
            processing_time = time.time() - start_time
            
            # Create processing frame
            frame = ProcessingFrame(
                frame_id=frame_data['frame_id'],
                timestamp=frame_data['timestamp'],
                rds_data=rds_data,
                peak_info=peak_info,
                targets=targets,
                processing_time=processing_time,
                memory_usage=memory_usage
            )
            
            # Add to buffer
            self.frame_buffer.add_frame(frame)
            
            # Update performance metrics
            self.processing_times.append(processing_time)
            self.memory_usage_history.append(memory_usage)
            
            logger.debug(f"Processed frame {frame_data['frame_id']} in {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_data['frame_id']}: {e}")
    
    def _simulate_target_processing(self, 
                                   rds_data: np.ndarray,
                                   peak_info: Dict) -> List[Dict]:
        """Simulate target processing (replace with actual processing)."""
        # This is a placeholder - replace with actual target processing
        targets = []
        for i, peak in enumerate(peak_info['peaks'][:10]):  # Limit to 10 targets for simulation
            target = {
                'target_id': f"target_{i}",
                'range_m': peak['range_m'],
                'doppler_hz': peak['doppler_hz'],
                'power_db': peak['power_db'],
                'azimuth_deg': np.random.uniform(-90, 90),
                'confidence': np.random.uniform(0.5, 1.0)
            }
            targets.append(target)
        
        return targets
    
    def get_latest_results(self, n_frames: int = 2) -> List[ProcessingFrame]:
        """Get latest processing results."""
        return self.frame_buffer.get_latest_frames(n_frames)
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics."""
        if not self.processing_times:
            return {
                'avg_processing_time': 0.0,
                'max_processing_time': 0.0,
                'min_processing_time': 0.0,
                'avg_memory_usage': 0.0,
                'max_memory_usage': 0.0,
                'frames_processed': 0
            }
        
        return {
            'avg_processing_time': np.mean(self.processing_times),
            'max_processing_time': np.max(self.processing_times),
            'min_processing_time': np.min(self.processing_times),
            'avg_memory_usage': np.mean(self.memory_usage_history),
            'max_memory_usage': np.max(self.memory_usage_history),
            'frames_processed': len(self.processing_times),
            'queue_size': self.processing_queue.qsize(),
            'buffer_size': len(self.frame_buffer.frames),
            'buffer_memory_mb': self.frame_buffer.get_memory_usage()
        }
    
    def optimize_memory_usage(self) -> None:
        """Optimize memory usage."""
        # Cleanup old frames
        self.frame_buffer.cleanup_old_frames()
        
        # Force garbage collection
        gc.collect()
        
        # Log memory optimization
        if psutil:
            current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            logger.info(f"Memory optimization complete. Current usage: {current_memory:.1f} MB")
        else:
            logger.info("Memory optimization complete.")
    
    def get_system_status(self) -> Dict:
        """Get system status information."""
        if psutil:
            cpu_percent = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            disk_info = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_total_gb': memory_info.total / (1024**3),
                'memory_available_gb': memory_info.available / (1024**3),
                'memory_percent': memory_info.percent,
                'disk_free_gb': disk_info.free / (1024**3),
                'disk_percent': (disk_info.used / disk_info.total) * 100,
                'processing_metrics': self.get_performance_metrics()
            }
        else:
            return {
                'cpu_percent': 0.0,
                'memory_total_gb': 0.0,
                'memory_available_gb': 0.0,
                'memory_percent': 0.0,
                'disk_free_gb': 0.0,
                'disk_percent': 0.0,
                'processing_metrics': self.get_performance_metrics()
            }

class RealTimeVelocityEstimator:
    """
    Real-time velocity estimator with optimizations.
    """
    
    def __init__(self,
                 radar_params: Dict,
                 frame_buffer_size: int = 10,
                 use_parallel: bool = True):
        """
        Initialize real-time velocity estimator.
        
        Args:
            radar_params: Radar parameters
            frame_buffer_size: Size of frame buffer
            use_parallel: Whether to use parallel processing
        """
        self.radar_params = radar_params
        self.real_time_processor = RealTimeProcessor(
            frame_buffer_size=frame_buffer_size,
            use_parallel=use_parallel
        )
        
        # Velocity estimation state
        self.velocity_history = deque(maxlen=20)
        self.angular_velocity_history = deque(maxlen=20)
        self.estimation_times = deque(maxlen=100)
        
        logger.info("Initialized real-time velocity estimator")
    
    def start_estimation(self) -> None:
        """Start real-time velocity estimation."""
        self.real_time_processor.start_processing()
        logger.info("Started real-time velocity estimation")
    
    def stop_estimation(self) -> None:
        """Stop real-time velocity estimation."""
        self.real_time_processor.stop_processing()
        logger.info("Stopped real-time velocity estimation")
    
    def add_frame(self, 
                  rds_data: np.ndarray,
                  peak_info: Dict,
                  frame_id: int = None) -> int:
        """
        Add frame for velocity estimation.
        
        Args:
            rds_data: RDS data
            peak_info: Peak information
            frame_id: Frame ID
            
        Returns:
            Frame ID
        """
        return self.real_time_processor.add_frame_for_processing(
            rds_data, peak_info, frame_id
        )
    
    def get_latest_velocity_estimate(self) -> Optional[Dict]:
        """Get latest velocity estimate."""
        latest_frames = self.real_time_processor.get_latest_results(2)
        
        if len(latest_frames) < 2:
            return None
        
        # This would contain actual velocity estimation logic
        # For now, return a placeholder
        return {
            'velocity': np.array([0.0, 0.0, 0.0]),
            'angular_velocity': np.array([0.0, 0.0, 0.0]),
            'confidence': 0.5,
            'timestamp': latest_frames[-1].timestamp,
            'frame_id': latest_frames[-1].frame_id
        }
    
    def get_estimation_statistics(self) -> Dict:
        """Get velocity estimation statistics."""
        metrics = self.real_time_processor.get_performance_metrics()
        
        return {
            'processing_metrics': metrics,
            'velocity_history_length': len(self.velocity_history),
            'angular_velocity_history_length': len(self.angular_velocity_history),
            'system_status': self.real_time_processor.get_system_status()
        }


def create_real_time_estimator(radar_params: Dict,
                              frame_buffer_size: int = 10,
                              use_parallel: bool = True) -> RealTimeVelocityEstimator:
    """
    Create a real-time velocity estimator.
    
    Args:
        radar_params: Radar parameters
        frame_buffer_size: Size of frame buffer
        use_parallel: Whether to use parallel processing
        
    Returns:
        Real-time velocity estimator
    """
    return RealTimeVelocityEstimator(radar_params, frame_buffer_size, use_parallel)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time velocity estimation')
    parser.add_argument('--rds', required=True, help='Path to RDS file')
    parser.add_argument('--peaks', required=True, help='Path to peak info file')
    parser.add_argument('--buffer-size', type=int, default=10, help='Frame buffer size')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    
    args = parser.parse_args()
    
    # Load data
    rds_data = np.load(args.rds)
    peak_data = np.load(args.peaks, allow_pickle=True)
    peak_info = dict(peak_data)
    
    # Create real-time estimator
    radar_params = {'fc': 77e9, 'lambda_c': 3e8 / 77e9, 'num_antennas': 8}
    estimator = create_real_time_estimator(radar_params, args.buffer_size, args.parallel)
    
    # Start estimation
    estimator.start_estimation()
    
    # Add frame for processing
    frame_id = estimator.add_frame(rds_data, peak_info)
    print(f"Added frame {frame_id} for processing")
    
    # Wait for processing
    time.sleep(2.0)
    
    # Get results
    velocity_estimate = estimator.get_latest_velocity_estimate()
    if velocity_estimate:
        print(f"Velocity estimate: {velocity_estimate}")
    
    # Get statistics
    stats = estimator.get_estimation_statistics()
    print(f"Estimation statistics: {stats}")
    
    # Stop estimation
    estimator.stop_estimation()
