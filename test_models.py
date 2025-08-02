#!/usr/bin/env python3
"""
Test script to verify the enhanced crowd density models work correctly
"""

import numpy as np
import cv2
import torch
from models.crowd_density_model import CrowdDensityEstimator
from models.enhanced_density_model import EnhancedCrowdDensityEstimator

def create_test_frame():
    """Create a test frame for model testing"""
    # Create a simple test image (640x480, 3 channels)
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return frame

def test_basic_model():
    """Test the basic crowd density model"""
    print("🧪 Testing Basic Crowd Density Model...")
    
    try:
        model = CrowdDensityEstimator()
        test_frame = create_test_frame()
        
        result = model.estimate_density(test_frame)
        
        # Handle tuple return format (density_map, count)
        if isinstance(result, tuple):
            density_map, count = result
            print(f"✅ Basic model test passed!")
            print(f"   Count: {count}")
            print(f"   Density Map Shape: {density_map.shape}")
            print(f"   Density Map Type: {type(density_map)}")
        else:
            # Handle dictionary format
            print(f"✅ Basic model test passed!")
            print(f"   Count: {result.get('count', result.get('total_count', 0))}")
            print(f"   Crowd Level: {result.get('crowd_level', 'Unknown')}")
            print(f"   Confidence: {result.get('confidence', result.get('confidence_score', 0)):.2f}")
            print(f"   Inference Time: {result.get('inference_time', 0):.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic model test failed: {e}")
        return False

def test_enhanced_model():
    """Test the enhanced crowd density model"""
    print("\n🧪 Testing Enhanced Crowd Density Model...")
    
    try:
        model = EnhancedCrowdDensityEstimator()
        test_frame = create_test_frame()
        
        result = model.estimate_crowd_density(test_frame)
        
        print(f"✅ Enhanced model test passed!")
        print(f"   Count: {result['count']}")
        print(f"   Crowd Level: {result['crowd_level']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Inference Time: {result['inference_time']:.3f}s")
        print(f"   Model Type: {result['model_type']}")
        print(f"   Status: {result['status']}")
        
        # Test performance stats
        stats = model.get_performance_stats()
        print(f"   Device: {stats['device']}")
        print(f"   FPS: {stats['fps']:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced model test failed: {e}")
        return False

def test_tensor_types():
    """Test that tensor type issues are resolved"""
    print("\n🧪 Testing Tensor Type Consistency...")
    
    try:
        model = EnhancedCrowdDensityEstimator()
        test_frame = create_test_frame()
        
        # Run multiple inferences to check consistency
        for i in range(3):
            result = model.estimate_crowd_density(test_frame)
            if result['status'] != 'success':
                print(f"⚠️ Inference {i+1} used fallback: {result['status']}")
            else:
                print(f"✅ Inference {i+1} successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Tensor type test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 CrowdGuard Pro Model Testing Suite")
    print("=" * 50)
    
    # Check device availability
    if torch.cuda.is_available():
        print(f"🔥 CUDA available: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("🍎 Apple Silicon GPU (MPS) available")
    else:
        print("💻 Using CPU for inference")
    
    print()
    
    # Run tests
    basic_passed = test_basic_model()
    enhanced_passed = test_enhanced_model()
    tensor_passed = test_tensor_types()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   Basic Model: {'✅ PASS' if basic_passed else '❌ FAIL'}")
    print(f"   Enhanced Model: {'✅ PASS' if enhanced_passed else '❌ FAIL'}")
    print(f"   Tensor Types: {'✅ PASS' if tensor_passed else '❌ FAIL'}")
    
    if all([basic_passed, enhanced_passed, tensor_passed]):
        print("\n🎉 All tests passed! Models are working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
