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
        """Build enhanced CNN model with proper initialization and data types"""
        
        class EnhancedDensityNet(nn.Module):
            def __init__(self):
                super(EnhancedDensityNet, self).__init__()
                
                # Initial feature extraction with proper initialization
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
                
                # Initialize weights properly
                self._initialize_weights()
            
            def _initialize_weights(self):
                """Initialize model weights to prevent type issues"""
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, 0, 0.01)
                        nn.init.constant_(m.bias, 0)
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
        """Enhanced preprocessing with proper data type handling"""
        try:
            # Resize frame
            resized = cv2.resize(frame, self.input_size)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Enhanced normalization with ImageNet statistics
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            
            # Ensure proper float32 conversion
            normalized = rgb_frame.astype(np.float32) / 255.0
            normalized = (normalized - mean) / std
            
            # Convert to tensor with explicit float32 type
            tensor = torch.from_numpy(normalized).float().permute(2, 0, 1).unsqueeze(0)
            
            return tensor
            
        except Exception as e:
            print(f"Error in frame preprocessing: {e}")
            # Return a zero tensor with correct shape and type
            return torch.zeros(1, 3, self.input_size[1], self.input_size[0], dtype=torch.float32)
    
    def estimate_density(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Enhanced density estimation with improved effectiveness and error handling
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (density_map, estimated_count)
        """
        try:
            # Validate input frame
            if frame is None or frame.size == 0:
                empty_map = np.zeros((224, 224), dtype=np.float32)
                return empty_map, 0
            
            # Preprocess frame with proper type handling
            input_tensor = self.preprocess_frame(frame)
            
            # Ensure model is in evaluation mode and proper device
            self.model.eval()
            
            # Run inference with gradient computation disabled
            with torch.no_grad():
                # Ensure input tensor is float32
                input_tensor = input_tensor.float()
                density_map = self.model(input_tensor)
                
                # Convert to numpy with proper type handling
                if isinstance(density_map, torch.Tensor):
                    density_map = density_map.squeeze().cpu().numpy().astype(np.float32)
                else:
                    density_map = np.array(density_map, dtype=np.float32)
            
            # Enhanced post-processing for better effectiveness
            density_map = np.maximum(density_map, 0)  # Ensure non-negative values
            
            # Apply adaptive filtering based on density values
            if np.max(density_map) > 0:
                # Normalize for better visualization
                density_map = density_map / np.max(density_map)
                
                # Apply intelligent smoothing
                kernel_size = 5 if np.mean(density_map) > 0.1 else 3
                density_map = cv2.GaussianBlur(density_map, (kernel_size, kernel_size), 1.0)
            
            # Resize density map to original frame size with proper interpolation
            original_height, original_width = frame.shape[:2]
            density_map_resized = cv2.resize(
                density_map, 
                (original_width, original_height),
                interpolation=cv2.INTER_CUBIC
            )
            
            # Improved crowd counting with multiple strategies
            crowd_count = self._enhanced_crowd_detection(frame, density_map_resized)
            
            # Ensure valid output types
            density_map_resized = density_map_resized.astype(np.float32)
            crowd_count = max(0, int(crowd_count))
            
            return density_map_resized, crowd_count
            
        except Exception as e:
            print(f"Error in density estimation: {e}")
            # Return safe fallback values
            try:
                empty_map = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
                return empty_map, 0
            except:
                # Ultimate fallback
                empty_map = np.zeros((224, 224), dtype=np.float32)
                return empty_map, 0
    
    def _enhanced_crowd_detection(self, frame: np.ndarray, density_map: np.ndarray) -> int:
        """
        Enhanced crowd detection with multiple algorithms and intelligent fusion
        """
        try:
            # Method 1: Density-based counting with adaptive thresholding
            density_sum = np.sum(density_map)
            density_mean = np.mean(density_map)
            density_std = np.std(density_map)
            
            # Adaptive scaling based on density distribution
            if density_std > 0:
                density_threshold = density_mean + 0.5 * density_std
                thresholded_map = np.where(density_map > density_threshold, density_map, 0)
                density_count = int(np.sum(thresholded_map) * 2.5)  # Scaling factor
            else:
                density_count = int(density_sum * 1.5)
            
            # Method 2: Traditional computer vision approach
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Apply CLAHE for better contrast
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                
                # Use adaptive threshold for better edge detection
                blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
                edges = cv2.Canny(blurred, 30, 100)
                
                # Morphological operations to connect broken edges
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
                
                # Find contours with hierarchy
                contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Intelligent contour filtering
                height, width = frame.shape[:2]
                min_area = max(200, (height * width) // 5000)  # Adaptive minimum area
                max_area = (height * width) // 20  # Adaptive maximum area
                
                valid_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if min_area < area < max_area:
                        # Additional shape analysis
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if 0.1 < circularity < 1.2:  # Filter out noise
                                valid_contours.append(contour)
                
                traditional_count = len(valid_contours)
                
            except Exception:
                traditional_count = 0
            
            # Method 3: Blob detection for person-like objects
            try:
                # Setup blob detector parameters
                params = cv2.SimpleBlobDetector_Params()
                params.filterByArea = True
                params.minArea = max(100, (frame.shape[0] * frame.shape[1]) // 10000)
                params.maxArea = (frame.shape[0] * frame.shape[1]) // 50
                params.filterByCircularity = False
                params.filterByConvexity = False
                params.filterByInertia = True
                params.minInertiaRatio = 0.2
                
                detector = cv2.SimpleBlobDetector_create(params)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                keypoints = detector.detect(gray)
                blob_count = len(keypoints)
                
            except Exception:
                blob_count = 0
            
            # Intelligent fusion of all methods
            counts = [density_count, traditional_count, blob_count]
            
            # Remove outliers (values too far from median)
            if len(counts) >= 3:
                median_count = np.median(counts)
                filtered_counts = [c for c in counts if abs(c - median_count) <= 2 * median_count + 1]
                if filtered_counts:
                    counts = filtered_counts
            
            # Weighted average based on confidence
            if density_sum > 0.01:  # High confidence in density map
                final_count = int(0.6 * density_count + 0.3 * traditional_count + 0.1 * blob_count)
            elif traditional_count > 0:  # Medium confidence in traditional methods
                final_count = int(0.3 * density_count + 0.5 * traditional_count + 0.2 * blob_count)
            else:  # Low confidence, use average
                final_count = int(np.mean(counts)) if counts else 0
            
            # Apply reasonable bounds
            max_reasonable_count = (frame.shape[0] * frame.shape[1]) // 2000  # Max people per pixel area
            final_count = min(final_count, max_reasonable_count)
            
            return max(0, final_count)
            
        except Exception as e:
            print(f"Error in enhanced crowd detection: {e}")
            # Fallback to simple density sum
            return max(0, int(np.sum(density_map) * 1.5))
    
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
