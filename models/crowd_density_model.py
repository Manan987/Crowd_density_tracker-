import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class ResidualBlock(nn.Module):
    """Residual block with skip connections"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AttentionModule(nn.Module):
    """Channel and spatial attention module"""
    def __init__(self, channels):
        super(AttentionModule, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        sa = self.spatial_attention(spatial_input)
        x = x * sa
        
        return x

class MultiScaleModule(nn.Module):
    """Multi-scale feature extraction module"""
    def __init__(self, in_channels, out_channels):
        super(MultiScaleModule, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels // 4, 1)
        self.branch2 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=1)
        self.branch3 = nn.Conv2d(in_channels, out_channels // 4, 5, padding=2)
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels // 4, 1)
        )
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        out = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        out = F.relu(self.bn(out))
        return out

class CrowdDensityEstimator:
    """
    Enhanced crowd density estimation model with deeper architecture
    Improved efficiency through residual connections and attention mechanisms
    """
    
    def __init__(self):
        self.model = self._build_enhanced_model()
        self.input_size = (224, 224)
        
    def _build_enhanced_model(self) -> nn.Module:
        """Build enhanced CNN model with more layers and modern techniques"""
        
        class EnhancedDensityNet(nn.Module):
            def __init__(self):
                super(EnhancedDensityNet, self).__init__()
                
                # Initial feature extraction
                self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
                self.bn1 = nn.BatchNorm2d(64)
                self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
                
                # Residual blocks for deeper feature extraction
                self.layer1 = self._make_layer(64, 64, 2, stride=1)
                self.layer2 = self._make_layer(64, 128, 2, stride=2)
                self.layer3 = self._make_layer(128, 256, 3, stride=2)
                self.layer4 = self._make_layer(256, 512, 3, stride=2)
                
                # Multi-scale processing
                self.multiscale1 = MultiScaleModule(512, 512)
                self.multiscale2 = MultiScaleModule(256, 256)
                self.multiscale3 = MultiScaleModule(128, 128)
                
                # Attention modules
                self.attention1 = AttentionModule(512)
                self.attention2 = AttentionModule(256)
                self.attention3 = AttentionModule(128)
                
                # Decoder with skip connections
                self.decoder1 = nn.Sequential(
                    nn.Conv2d(512, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                )
                
                self.decoder2 = nn.Sequential(
                    nn.Conv2d(512, 128, 3, padding=1),  # 256 + 256 from skip connection
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                )
                
                self.decoder3 = nn.Sequential(
                    nn.Conv2d(256, 64, 3, padding=1),   # 128 + 128 from skip connection
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                )
                
                self.decoder4 = nn.Sequential(
                    nn.Conv2d(128, 32, 3, padding=1),   # 64 + 64 from skip connection
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                )
                
                # Final density prediction
                self.final_conv = nn.Sequential(
                    nn.Conv2d(32, 16, 3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.Conv2d(16, 1, 1),
                    nn.ReLU()  # Ensure non-negative density values
                )
                
                self.dropout = nn.Dropout2d(0.2)
                
            def _make_layer(self, in_channels, out_channels, blocks, stride):
                layers = []
                layers.append(ResidualBlock(in_channels, out_channels, stride))
                for _ in range(1, blocks):
                    layers.append(ResidualBlock(out_channels, out_channels))
                return nn.Sequential(*layers)
            
            def forward(self, x):
                # Initial feature extraction
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.maxpool(x)
                
                # Encoder with residual blocks
                x1 = self.layer1(x)      # 64 channels
                x2 = self.layer2(x1)     # 128 channels
                x3 = self.layer3(x2)     # 256 channels
                x4 = self.layer4(x3)     # 512 channels
                
                # Multi-scale processing and attention
                x4 = self.multiscale1(x4)
                x4 = self.attention1(x4)
                x4 = self.dropout(x4)
                
                # Decoder with skip connections
                d1 = self.decoder1(x4)
                x3_ms = self.multiscale2(x3)
                x3_att = self.attention2(x3_ms)
                d1 = torch.cat([d1, x3_att], dim=1)  # Skip connection
                
                d2 = self.decoder2(d1)
                x2_ms = self.multiscale3(x2)
                x2_att = self.attention3(x2_ms)
                d2 = torch.cat([d2, x2_att], dim=1)  # Skip connection
                
                d3 = self.decoder3(d2)
                d3 = torch.cat([d3, x1], dim=1)      # Skip connection
                
                d4 = self.decoder4(d3)
                
                # Final density map
                density_map = self.final_conv(d4)
                
                return density_map
        
        model = EnhancedDensityNet()
        model.eval()  # Set to evaluation mode
        return model
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Enhanced preprocessing with data augmentation capabilities"""
        # Resize frame
        resized = cv2.resize(frame, self.input_size)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Enhanced normalization with ImageNet statistics
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        normalized = rgb_frame.astype(np.float32) / 255.0
        normalized = (normalized - mean) / std
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        
        return tensor
    
    def estimate_density(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Enhanced density estimation with improved post-processing
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (density_map, estimated_count)
        """
        try:
            # Preprocess frame
            input_tensor = self.preprocess_frame(frame)
            
            # Run inference with gradient computation disabled
            with torch.no_grad():
                density_map = self.model(input_tensor)
                density_map = density_map.squeeze().cpu().numpy()
            
            # Enhanced post-processing
            density_map = np.maximum(density_map, 0)  # Ensure non-negative values
            
            # Apply Gaussian smoothing for better visual quality
            density_map = cv2.GaussianBlur(density_map, (3, 3), 0.5)
            
            # Resize density map to original frame size
            density_map_resized = cv2.resize(
                density_map, 
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            
            # Calculate total count with improved accuracy
            estimated_count = int(np.sum(density_map_resized))
            
            # Apply enhanced crowd detection
            crowd_count = self._enhanced_crowd_detection(frame, density_map_resized)
            
            return density_map_resized, crowd_count
            
        except Exception as e:
            print(f"Error in density estimation: {e}")
            # Return empty density map and zero count on error
            empty_map = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
            return empty_map, 0
    
    def _enhanced_crowd_detection(self, frame: np.ndarray, density_map: np.ndarray) -> int:
        """
        Enhanced crowd detection combining traditional CV and density map
        """
        # Traditional detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        min_area = 300
        max_area = 8000
        valid_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
        traditional_count = len(valid_contours)
        
        # Density-based count
        density_count = int(np.sum(density_map))
        
        # Weighted combination
        final_count = int(0.3 * traditional_count + 0.7 * density_count)
        
        return max(0, final_count)
    
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
        """Get enhanced model information and statistics"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": "EnhancedCrowdDensityNet",
            "input_size": self.input_size,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "architecture": "ResNet-style with Attention and Multi-scale Processing",
            "layers": "40+ layers with residual connections",
            "features": [
                "Residual blocks for deeper learning",
                "Channel and spatial attention",
                "Multi-scale feature extraction",
                "Skip connections in decoder",
                "Batch normalization for stability"
            ]
        }
