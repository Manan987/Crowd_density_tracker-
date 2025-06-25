import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Dict, List

class DilatedConvBlock(nn.Module):
    """Dilated convolution block for multi-scale feature extraction"""
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2, 3, 6]):
        super(DilatedConvBlock, self).__init__()
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels // len(dilation_rates), 3, 
                     padding=rate, dilation=rate) for rate in dilation_rates
        ])
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        outputs = [conv(x) for conv in self.dilated_convs]
        out = torch.cat(outputs, dim=1)
        return self.relu(self.bn(out))

class CSRNetBackbone(nn.Module):
    """CSRNet-inspired backbone for crowd density estimation"""
    def __init__(self):
        super(CSRNetBackbone, self).__init__()
        
        # Frontend: VGG-like feature extractor
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
        )
        
        # Backend: Dilated convolutions for density map generation
        self.backend = nn.Sequential(
            DilatedConvBlock(512, 512),
            DilatedConvBlock(512, 512),
            DilatedConvBlock(512, 512),
            DilatedConvBlock(512, 256),
            nn.Conv2d(256, 1, 1)
        )
        
    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x

class ScaleAdaptiveModule(nn.Module):
    """Scale-adaptive module for handling varying crowd densities"""
    def __init__(self, in_channels):
        super(ScaleAdaptiveModule, self).__init__()
        self.scale_branches = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.Conv2d(in_channels, in_channels // 4, 5, padding=2),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels // 4, 1),
                nn.Upsample(scale_factor=None, mode='bilinear')
            )
        ])
        
    def forward(self, x):
        h, w = x.shape[2:]
        branch_outputs = []
        
        for i, branch in enumerate(self.scale_branches):
            if i == 3:  # Global pooling branch
                out = branch[:-1](x)  # Apply pooling and conv
                out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
            else:
                out = branch(x)
            branch_outputs.append(out)
            
        return torch.cat(branch_outputs, dim=1)

class EnhancedCrowdDensityEstimator:
    """Production-ready crowd density estimator with advanced features"""
    
    def __init__(self, model_type='csrnet', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_type = model_type
        self.model = self._build_model().to(device)
        self.input_size = (512, 512)  # Higher resolution for better accuracy
        
        # Performance monitoring
        self.inference_times = []
        self.accuracy_metrics = {'mae': [], 'mse': []}
        
    def _build_model(self):
        """Build the selected model architecture"""
        if self.model_type == 'csrnet':
            return CSRNetBackbone()
        else:
            # Fallback to basic model
            return self._build_basic_model()
    
    def _build_basic_model(self):
        """Fallback basic model"""
        class BasicDensityNet(nn.Module):
            def __init__(self):
                super(BasicDensityNet, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2), 
                    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(128, 1, 1)
                )
            
            def forward(self, x):
                return self.features(x)
        
        return BasicDensityNet()
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Advanced preprocessing with augmentation support"""
        # Resize maintaining aspect ratio
        h, w = frame.shape[:2]
        target_h, target_w = self.input_size
        
        # Calculate scaling factor
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize and pad
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Create padded image
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Convert and normalize
        rgb_frame = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb_frame.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        
        # Convert to tensor
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)
    
    def estimate_density(self, frame: np.ndarray) -> Dict:
        """Enhanced density estimation with comprehensive output"""
        import time
        start_time = time.time()
        
        try:
            # Preprocess
            input_tensor = self.preprocess_frame(frame)
            
            # Inference
            with torch.no_grad():
                density_map = self.model(input_tensor)
                density_map = density_map.squeeze().cpu().numpy()
            
            # Post-process
            density_map = np.maximum(density_map, 0)
            
            # Resize to original frame size
            original_h, original_w = frame.shape[:2]
            density_map_resized = cv2.resize(
                density_map, (original_w, original_h), 
                interpolation=cv2.INTER_LINEAR
            )
            
            # Calculate metrics
            total_count = int(np.sum(density_map_resized))
            max_density = np.max(density_map_resized)
            avg_density = np.mean(density_map_resized)
            
            # Performance tracking
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            return {
                'density_map': density_map_resized,
                'total_count': total_count,
                'max_density': float(max_density),
                'avg_density': float(avg_density),
                'inference_time': inference_time,
                'confidence_score': self._calculate_confidence(density_map_resized),
                'risk_level': self._assess_risk_level(total_count, max_density)
            }
            
        except Exception as e:
            print(f"Error in density estimation: {e}")
            return self._get_empty_result(frame.shape[:2])
    
    def _calculate_confidence(self, density_map: np.ndarray) -> float:
        """Calculate confidence score based on density map characteristics"""
        # Simple confidence based on density distribution
        std_dev = np.std(density_map)
        mean_density = np.mean(density_map)
        
        # Higher confidence for more uniform distributions
        if mean_density > 0:
            confidence = 1.0 / (1.0 + std_dev / mean_density)
        else:
            confidence = 0.5
            
        return min(max(confidence, 0.0), 1.0)
    
    def _assess_risk_level(self, total_count: int, max_density: float) -> str:
        """Assess risk level based on crowd metrics"""
        if total_count > 100 or max_density > 0.8:
            return 'HIGH'
        elif total_count > 50 or max_density > 0.5:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _get_empty_result(self, shape: Tuple[int, int]) -> Dict:
        """Return empty result on error"""
        return {
            'density_map': np.zeros(shape, dtype=np.float32),
            'total_count': 0,
            'max_density': 0.0,
            'avg_density': 0.0,
            'inference_time': 0.0,
            'confidence_score': 0.0,
            'risk_level': 'UNKNOWN'
        }
    
    def get_performance_stats(self) -> Dict:
        """Get model performance statistics"""
        if not self.inference_times:
            return {'avg_inference_time': 0, 'fps': 0}
            
        avg_time = np.mean(self.inference_times[-100:])  # Last 100 frames
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'avg_inference_time': avg_time,
            'fps': fps,
            'total_frames_processed': len(self.inference_times)
        }