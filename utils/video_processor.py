import cv2
import numpy as np
from typing import Tuple, Optional, List
import threading
import queue
import time

class VideoProcessor:
    """
    Video processing utilities for crowd density monitoring
    Handles video streams, frame processing, and visualization
    """
    
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=10)
        self.processing_thread = None
        self.is_processing = False
        
    def create_advanced_overlay(self, frame: np.ndarray, density_map: np.ndarray, 
                              alpha: float = 0.6, show_zones: bool = True, 
                              show_flow: bool = False) -> np.ndarray:
        """Create advanced overlay with multiple visualization layers"""
        try:
            # Start with basic heatmap
            result_frame = self.create_heatmap_overlay(frame, density_map, alpha)
            
            # Add density zones if requested
            if show_zones:
                result_frame = self.add_density_zones_overlay(result_frame)
            
            # Add flow vectors if requested
            if show_flow:
                result_frame = self.add_flow_vectors_overlay(result_frame)
            
            # Add advanced info panel
            result_frame = self.add_advanced_info_panel(result_frame, density_map)
            
            return result_frame
            
        except Exception as e:
            print(f"Error creating advanced overlay: {e}")
            return frame
    
    def create_heatmap_overlay(self, frame: np.ndarray, density_map: np.ndarray, 
                             alpha: float = 0.6) -> np.ndarray:
        """
        Create heatmap overlay on original frame
        
        Args:
            frame: Original video frame
            density_map: Crowd density map
            alpha: Transparency factor for overlay
            
        Returns:
            Frame with heatmap overlay
        """
        try:
            # Ensure density map has the same dimensions as frame
            if density_map.shape[:2] != frame.shape[:2]:
                density_map = cv2.resize(density_map, (frame.shape[1], frame.shape[0]))
            
            # Normalize density map to 0-255 range
            density_normalized = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
            density_normalized = density_normalized.astype(np.uint8)
            
            # Apply colormap (red = high density, blue = low density)
            heatmap = cv2.applyColorMap(density_normalized, cv2.COLORMAP_JET)
            
            # Create overlay
            overlay = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
            
            # Add density information text
            self._add_density_info(overlay, density_map)
            
            return overlay
            
        except Exception as e:
            print(f"Error creating heatmap overlay: {e}")
            return frame
    
    def _add_density_info(self, frame: np.ndarray, density_map: np.ndarray):
        """Add density information text to frame"""
        try:
            # Calculate statistics
            total_density = np.sum(density_map)
            max_density = np.max(density_map)
            avg_density = np.mean(density_map)
            
            # Add text overlays
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            color = (255, 255, 255)
            thickness = 2
            
            # Background rectangles for better text visibility
            cv2.rectangle(frame, (10, 10), (350, 100), (0, 0, 0), -1)
            
            # Text information
            cv2.putText(frame, f"Total Count: {int(total_density)}", 
                       (20, 35), font, font_scale, color, thickness)
            cv2.putText(frame, f"Max Density: {max_density:.2f}", 
                       (20, 60), font, font_scale, color, thickness)
            cv2.putText(frame, f"Avg Density: {avg_density:.2f}", 
                       (20, 85), font, font_scale, color, thickness)
            
        except Exception as e:
            print(f"Error adding density info: {e}")
    
    def add_alert_indicators(self, frame: np.ndarray, alert_level: str, 
                           density_value: float) -> np.ndarray:
        """
        Add visual alert indicators to frame
        
        Args:
            frame: Input frame
            alert_level: Current alert level (Normal, Warning, Critical)
            density_value: Current density value
            
        Returns:
            Frame with alert indicators
        """
        try:
            h, w = frame.shape[:2]
            
            # Alert colors
            alert_colors = {
                "Normal": (0, 255, 0),      # Green
                "Warning": (0, 255, 255),   # Yellow
                "Critical": (0, 0, 255)     # Red
            }
            
            color = alert_colors.get(alert_level, (128, 128, 128))
            
            # Draw border based on alert level
            border_thickness = 5 if alert_level != "Normal" else 2
            cv2.rectangle(frame, (0, 0), (w-1, h-1), color, border_thickness)
            
            # Add alert text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            
            # Alert status background
            text_size = cv2.getTextSize(f"{alert_level} ALERT", font, font_scale, thickness)[0]
            cv2.rectangle(frame, (w - text_size[0] - 20, 10), 
                         (w - 10, text_size[1] + 30), color, -1)
            
            # Alert text
            cv2.putText(frame, f"{alert_level} ALERT", 
                       (w - text_size[0] - 15, text_size[1] + 20), 
                       font, font_scale, (255, 255, 255), thickness)
            
            # Density value
            cv2.putText(frame, f"{density_value:.1f} people/m²", 
                       (w - text_size[0] - 15, text_size[1] + 50), 
                       font, 0.6, (255, 255, 255), 1)
            
            # Blinking effect for critical alerts
            if alert_level == "Critical":
                current_time = time.time()
                if int(current_time * 2) % 2:  # Blink every 0.5 seconds
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                    frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
            
            return frame
            
        except Exception as e:
            print(f"Error adding alert indicators: {e}")
            return frame
    
    def add_grid_overlay(self, frame: np.ndarray, grid_size: int = 50) -> np.ndarray:
        """
        Add grid overlay to help visualize density distribution
        
        Args:
            frame: Input frame
            grid_size: Size of grid squares in pixels
            
        Returns:
            Frame with grid overlay
        """
        try:
            h, w = frame.shape[:2]
            color = (128, 128, 128)
            thickness = 1
            
            # Vertical lines
            for x in range(0, w, grid_size):
                cv2.line(frame, (x, 0), (x, h), color, thickness)
            
            # Horizontal lines
            for y in range(0, h, grid_size):
                cv2.line(frame, (0, y), (w, y), color, thickness)
            
            return frame
            
        except Exception as e:
            print(f"Error adding grid overlay: {e}")
            return frame
    
    def detect_motion_areas(self, current_frame: np.ndarray, 
                          previous_frame: Optional[np.ndarray]) -> np.ndarray:
        """
        Detect motion areas between consecutive frames
        
        Args:
            current_frame: Current video frame
            previous_frame: Previous video frame
            
        Returns:
            Motion mask
        """
        if previous_frame is None:
            return np.zeros(current_frame.shape[:2], dtype=np.uint8)
        
        try:
            # Convert to grayscale
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(current_gray, previous_gray)
            
            # Apply threshold
            _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
            
            return motion_mask
            
        except Exception as e:
            print(f"Error detecting motion areas: {e}")
            return np.zeros(current_frame.shape[:2], dtype=np.uint8)
    
    def create_density_zones(self, frame_shape: Tuple[int, int], 
                           zones: List[dict]) -> np.ndarray:
        """
        Create density zones for focused monitoring
        
        Args:
            frame_shape: Shape of the frame (height, width)
            zones: List of zone dictionaries with coordinates and labels
            
        Returns:
            Zone mask
        """
        try:
            h, w = frame_shape
            zone_mask = np.zeros((h, w), dtype=np.uint8)
            
            for i, zone in enumerate(zones):
                # Extract zone coordinates
                x1, y1, x2, y2 = zone.get('coords', [0, 0, w//2, h//2])
                
                # Draw zone
                cv2.rectangle(zone_mask, (x1, y1), (x2, y2), i+1, -1)
            
            return zone_mask
            
        except Exception as e:
            print(f"Error creating density zones: {e}")
            return np.zeros(frame_shape, dtype=np.uint8)
    
    def calculate_zone_densities(self, density_map: np.ndarray, 
                               zone_mask: np.ndarray) -> dict:
        """
        Calculate density values for specific zones
        
        Args:
            density_map: Overall density map
            zone_mask: Zone mask with different values for each zone
            
        Returns:
            Dictionary with zone densities
        """
        try:
            zone_densities = {}
            unique_zones = np.unique(zone_mask)
            
            for zone_id in unique_zones:
                if zone_id == 0:  # Skip background
                    continue
                
                zone_pixels = zone_mask == zone_id
                zone_density = np.sum(density_map[zone_pixels])
                zone_densities[f"Zone_{zone_id}"] = zone_density
            
            return zone_densities
            
        except Exception as e:
            print(f"Error calculating zone densities: {e}")
            return {}
    
    def enhance_frame_quality(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance frame quality for better processing
        
        Args:
            frame: Input frame
            
        Returns:
            Enhanced frame
        """
        try:
            # Apply histogram equalization to improve contrast
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Apply slight Gaussian blur to reduce noise
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            return enhanced
            
        except Exception as e:
            print(f"Error enhancing frame quality: {e}")
            return frame
    
    def add_density_zones_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add density zone overlays to the frame"""
        try:
            h, w = frame.shape[:2]
            
            # Define monitoring zones
            zones = [
                {"name": "Entry Zone", "coords": (50, 50, 200, 150), "color": (0, 255, 255)},
                {"name": "Main Area", "coords": (250, 100, 500, 300), "color": (255, 255, 0)},
                {"name": "Exit Zone", "coords": (w-200, h-150, w-50, h-50), "color": (255, 0, 255)}
            ]
            
            for zone in zones:
                x1, y1, x2, y2 = zone["coords"]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                
                # Draw zone rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), zone["color"], 2)
                
                # Add zone label with background
                label = zone["name"]
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1-text_h-10), (x1+text_w+10, y1), zone["color"], -1)
                cv2.putText(frame, label, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            return frame
            
        except Exception as e:
            print(f"Error adding density zones: {e}")
            return frame
    
    def add_flow_vectors_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add crowd flow vector overlays"""
        try:
            h, w = frame.shape[:2]
            
            # Generate sample flow vectors (in real implementation, these would come from optical flow)
            grid_size = 60
            for y in range(grid_size, h-grid_size, grid_size):
                for x in range(grid_size, w-grid_size, grid_size):
                    # Simulate flow towards exits
                    flow_x = np.random.normal(0, 20)
                    flow_y = np.random.normal(0, 20)
                    
                    # Draw flow arrow
                    end_x = int(x + flow_x)
                    end_y = int(y + flow_y)
                    
                    cv2.arrowedLine(frame, (x, y), (end_x, end_y), (0, 255, 0), 2, tipLength=0.3)
            
            return frame
            
        except Exception as e:
            print(f"Error adding flow vectors: {e}")
            return frame
    
    def add_advanced_info_panel(self, frame: np.ndarray, density_map: np.ndarray) -> np.ndarray:
        """Add advanced information panel with comprehensive stats"""
        try:
            h, w = frame.shape[:2]
            
            # Calculate advanced statistics
            total_density = np.sum(density_map)
            max_density = np.max(density_map)
            avg_density = np.mean(density_map)
            density_variance = np.var(density_map)
            
            # Hotspot detection
            hotspot_threshold = avg_density + 2 * np.sqrt(density_variance)
            hotspot_count = np.sum(density_map > hotspot_threshold)
            
            # Create info panel background
            panel_width = 300
            panel_height = 180
            panel_x = w - panel_width - 10
            panel_y = 10
            
            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Border
            cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                         (255, 255, 255), 2)
            
            # Add title
            cv2.putText(frame, "CROWD ANALYTICS", (panel_x + 10, panel_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add statistics
            stats_text = [
                f"Total Count: {int(total_density)}",
                f"Peak Density: {max_density:.2f}",
                f"Avg Density: {avg_density:.2f}",
                f"Variance: {density_variance:.2f}",
                f"Hotspots: {hotspot_count}",
                f"Risk Level: {'HIGH' if max_density > 3.0 else 'MEDIUM' if max_density > 1.5 else 'LOW'}"
            ]
            
            for i, text in enumerate(stats_text):
                y_pos = panel_y + 50 + (i * 20)
                color = (0, 255, 0) if i == len(stats_text)-1 and max_density < 1.5 else \
                        (0, 255, 255) if i == len(stats_text)-1 and max_density < 3.0 else \
                        (0, 0, 255) if i == len(stats_text)-1 else (255, 255, 255)
                
                cv2.putText(frame, text, (panel_x + 10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, f"Updated: {timestamp}", (panel_x + 10, panel_y + panel_height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            return frame
            
        except Exception as e:
            print(f"Error adding info panel: {e}")
            return frame
    
    def detect_crowd_patterns(self, frame: np.ndarray) -> dict:
        """Detect and analyze crowd movement patterns"""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (21, 21), 0)
            
            # Detect edges for crowd boundaries
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours representing crowd clusters
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze crowd clusters
            clusters = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Filter small noise
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    density_estimate = area / (w * h) if (w * h) > 0 else 0
                    
                    clusters.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'density': density_estimate,
                        'center': (x + w//2, y + h//2)
                    })
            
            # Calculate movement vectors between clusters
            movement_vectors = []
            if len(clusters) >= 2:
                for i in range(len(clusters)-1):
                    c1, c2 = clusters[i], clusters[i+1]
                    vector = (c2['center'][0] - c1['center'][0], c2['center'][1] - c1['center'][1])
                    movement_vectors.append(vector)
            
            return {
                'cluster_count': len(clusters),
                'total_crowd_area': sum(c['area'] for c in clusters),
                'average_cluster_size': np.mean([c['area'] for c in clusters]) if clusters else 0,
                'movement_vectors': movement_vectors,
                'crowd_dispersion': np.std([c['center'][0] for c in clusters]) if clusters else 0
            }
            
        except Exception as e:
            print(f"Error detecting crowd patterns: {e}")
            return {}
    
    def add_safety_zone_indicators(self, frame: np.ndarray, safety_zones: list) -> np.ndarray:
        """Add safety zone indicators to the frame"""
        try:
            for zone in safety_zones:
                x, y, w, h = zone.get('bounds', (0, 0, 100, 100))
                status = zone.get('status', 'safe')
                
                # Color coding for safety status
                color_map = {
                    'safe': (0, 255, 0),      # Green
                    'caution': (0, 255, 255), # Yellow
                    'danger': (0, 0, 255)     # Red
                }
                
                color = color_map.get(status, (128, 128, 128))
                
                # Draw zone boundary
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                
                # Add status indicator
                cv2.circle(frame, (x + w - 20, y + 20), 15, color, -1)
                cv2.putText(frame, status.upper()[:1], (x + w - 25, y + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            print(f"Error adding safety zones: {e}")
            return frame
    
    def get_processing_stats(self) -> dict:
        """Get video processing statistics"""
        return {
            "queue_size": self.frame_queue.qsize(),
            "is_processing": self.is_processing,
            "max_queue_size": self.frame_queue.maxsize
        }
