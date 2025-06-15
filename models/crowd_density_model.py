import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class CrowdDensityEstimator:
    """
    Crowd density estimation model using CNN architecture
    Simplified implementation for real-time processing
    """
    
    def __init__(self):
        self.model = self._build_model()
        self.input_size = (224, 224)
        
    def _build_model(self) -> nn.Module:
        """Build a simplified CNN model for crowd density estimation"""
        
        class DensityNet(nn.Module):
            def __init__(self):
                super(DensityNet, self).__init__()
                
                # Feature extraction layers
                self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
                
                # Density regression layers
                self.conv5 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
                self.conv6 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
                self.conv7 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
                self.conv8 = nn.Conv2d(32, 1, kernel_size=1)
                
                self.pool = nn.MaxPool2d(2, 2)
                self.dropout = nn.Dropout(0.3)
                
            def forward(self, x):
                # Feature extraction
                x = F.relu(self.conv1(x))
                x = self.pool(x)
                x = F.relu(self.conv2(x))
                x = self.pool(x)
                x = F.relu(self.conv3(x))
                x = self.pool(x)
                x = F.relu(self.conv4(x))
                
                # Density map generation
                x = F.relu(self.conv5(x))
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                x = F.relu(self.conv6(x))
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                x = F.relu(self.conv7(x))
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                x = self.conv8(x)
                
                return x
        
        model = DensityNet()
        model.eval()  # Set to evaluation mode
        return model
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for model input"""
        # Resize frame
        resized = cv2.resize(frame, self.input_size)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        normalized = rgb_frame.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        
        return tensor
    
    def estimate_density(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Estimate crowd density from input frame
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (density_map, estimated_count)
        """
        try:
            # Preprocess frame
            input_tensor = self.preprocess_frame(frame)
            
            # Run inference
            with torch.no_grad():
                density_map = self.model(input_tensor)
                density_map = density_map.squeeze().cpu().numpy()
            
            # Apply post-processing
            density_map = np.maximum(density_map, 0)  # Ensure non-negative values
            
            # Resize density map to original frame size
            density_map_resized = cv2.resize(
                density_map, 
                (frame.shape[1], frame.shape[0])
            )
            
            # Calculate total count
            estimated_count = int(np.sum(density_map_resized))
            
            # Apply synthetic crowd detection (for demo purposes)
            # In a real implementation, this would be replaced by actual trained model inference
            crowd_count = self._synthetic_crowd_detection(frame)
            
            return density_map_resized, crowd_count
            
        except Exception as e:
            print(f"Error in density estimation: {e}")
            # Return empty density map and zero count on error
            empty_map = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
            return empty_map, 0
    
    def _synthetic_crowd_detection(self, frame: np.ndarray) -> int:
        """
        Synthetic crowd detection for demonstration purposes
        In production, this would be replaced by actual ML model inference
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours (approximating people/objects)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area (approximate person size)
        min_area = 500
        max_area = 5000
        valid_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
        
        # Add some randomness to simulate varying detection accuracy
        import random
        base_count = len(valid_contours)
        noise_factor = random.uniform(0.8, 1.2)
        
        return max(0, int(base_count * noise_factor))
    
    def generate_density_heatmap(self, density_map: np.ndarray) -> np.ndarray:
        """Generate a visual heatmap from density map"""
        # Normalize density map
        normalized = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        return heatmap
    
    def update_model_weights(self, model_path: str):
        """Load updated model weights (for model updates)"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint)
            print("Model weights updated successfully")
        except Exception as e:
            print(f"Error loading model weights: {e}")
    
    def get_model_info(self) -> dict:
        """Get model information and statistics"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": "CrowdDensityNet",
            "input_size": self.input_size,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "architecture": "CNN-based density regression"
        }
