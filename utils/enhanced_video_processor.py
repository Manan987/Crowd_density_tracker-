import cv2
import numpy as np
import threading
import queue
import time
from typing import Dict, List, Callable, Optional
from concurrent.futures import ThreadPoolExecutor
import logging

class MultiStreamProcessor:
    """Scalable multi-stream video processor for real-time crowd monitoring"""
    
    def __init__(self, max_streams: int = 4, buffer_size: int = 10):
        self.max_streams = max_streams
        self.buffer_size = buffer_size
        self.streams = {}
        self.frame_queues = {}
        self.result_queues = {}
        self.processing_threads = {}
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=max_streams)
        
        # Performance monitoring
        self.stream_stats = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def add_stream(self, stream_id: str, source: str, 
                   density_estimator, alert_callback: Optional[Callable] = None) -> bool:
        """Add a new video stream for processing"""
        if len(self.streams) >= self.max_streams:
            self.logger.warning(f"Maximum streams ({self.max_streams}) reached")
            return False
        
        try:
            # Initialize video capture
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                self.logger.error(f"Failed to open stream: {source}")
                return False
            
            # Setup stream components
            self.streams[stream_id] = {
                'capture': cap,
                'source': source,
                'density_estimator': density_estimator,
                'alert_callback': alert_callback,
                'last_frame_time': time.time(),
                'frame_count': 0
            }
            
            self.frame_queues[stream_id] = queue.Queue(maxsize=self.buffer_size)
            self.result_queues[stream_id] = queue.Queue(maxsize=self.buffer_size)
            
            # Initialize stats
            self.stream_stats[stream_id] = {
                'fps': 0,
                'processing_time': 0,
                'total_frames': 0,
                'dropped_frames': 0,
                'last_result': None
            }
            
            self.logger.info(f"Stream {stream_id} added successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding stream {stream_id}: {e}")
            return False
    
    def start_processing(self):
        """Start processing all streams"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start capture threads for each stream
        for stream_id in self.streams:
            capture_thread = threading.Thread(
                target=self._capture_frames, 
                args=(stream_id,),
                daemon=True
            )
            capture_thread.start()
            
            # Start processing thread
            process_thread = threading.Thread(
                target=self._process_stream,
                args=(stream_id,),
                daemon=True
            )
            process_thread.start()
            
            self.processing_threads[stream_id] = {
                'capture': capture_thread,
                'process': process_thread
            }
        
        self.logger.info("All streams started")
    
    def _capture_frames(self, stream_id: str):
        """Capture frames from video stream"""
        stream = self.streams[stream_id]
        cap = stream['capture']
        frame_queue = self.frame_queues[stream_id]
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                self.logger.warning(f"Failed to read frame from {stream_id}")
                time.sleep(0.1)
                continue
            
            # Add timestamp
            timestamp = time.time()
            
            try:
                # Non-blocking put, drop frame if queue is full
                frame_queue.put_nowait((frame, timestamp))
                stream['frame_count'] += 1
            except queue.Full:
                # Drop frame if queue is full
                self.stream_stats[stream_id]['dropped_frames'] += 1
    
    def _process_stream(self, stream_id: str):
        """Process frames for density estimation"""
        stream = self.streams[stream_id]
        frame_queue = self.frame_queues[stream_id]
        result_queue = self.result_queues[stream_id]
        density_estimator = stream['density_estimator']
        alert_callback = stream['alert_callback']
        
        while self.is_running:
            try:
                # Get frame with timeout
                frame, timestamp = frame_queue.get(timeout=1.0)
                
                # Process frame
                start_time = time.time()
                result = density_estimator.estimate_density(frame)
                processing_time = time.time() - start_time
                
                # Add metadata
                result.update({
                    'stream_id': stream_id,
                    'timestamp': timestamp,
                    'processing_time': processing_time,
                    'frame': frame
                })
                
                # Update stats
                stats = self.stream_stats[stream_id]
                stats['processing_time'] = processing_time
                stats['total_frames'] += 1
                stats['last_result'] = result
                
                # Calculate FPS
                current_time = time.time()
                if stream['last_frame_time'] > 0:
                    fps = 1.0 / (current_time - stream['last_frame_time'])
                    stats['fps'] = fps
                stream['last_frame_time'] = current_time
                
                # Store result
                try:
                    result_queue.put_nowait(result)
                except queue.Full:
                    # Remove oldest result if queue is full
                    try:
                        result_queue.get_nowait()
                        result_queue.put_nowait(result)
                    except queue.Empty:
                        pass
                
                # Trigger alert if callback provided
                if alert_callback and result['risk_level'] == 'HIGH':
                    alert_callback(stream_id, result)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing stream {stream_id}: {e}")
    
    def get_latest_results(self) -> Dict:
        """Get latest results from all streams"""
        results = {}
        
        for stream_id in self.streams:
            try:
                # Get most recent result
                result = self.result_queues[stream_id].get_nowait()
                results[stream_id] = result
            except queue.Empty:
                # Use last known result if available
                if self.stream_stats[stream_id]['last_result']:
                    results[stream_id] = self.stream_stats[stream_id]['last_result']
        
        return results
    
    def get_stream_stats(self) -> Dict:
        """Get performance statistics for all streams"""
        return self.stream_stats.copy()
    
    def stop_processing(self):
        """Stop all stream processing"""
        self.is_running = False
        
        # Close all video captures
        for stream in self.streams.values():
            stream['capture'].release()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("All streams stopped")
    
    def remove_stream(self, stream_id: str) -> bool:
        """Remove a stream from processing"""
        if stream_id not in self.streams:
            return False
        
        # Close capture
        self.streams[stream_id]['capture'].release()
        
        # Remove from dictionaries
        del self.streams[stream_id]
        del self.frame_queues[stream_id]
        del self.result_queues[stream_id]
        del self.stream_stats[stream_id]
        
        if stream_id in self.processing_threads:
            del self.processing_threads[stream_id]
        
        self.logger.info(f"Stream {stream_id} removed")
        return True