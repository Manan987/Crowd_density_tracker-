#!/usr/bin/env python3
"""
Simple test to verify Streamlit compatibility fixes
"""

import streamlit as st
import numpy as np

def test_image_display():
    """Test the image display function"""
    print("Testing image display compatibility...")
    
    try:
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test our safe_image_display function
        from app import safe_image_display
        
        # Create a mock placeholder
        class MockPlaceholder:
            def image(self, *args, **kwargs):
                print(f"✅ Image called with args: {len(args)}, kwargs: {list(kwargs.keys())}")
                return True
            
            def warning(self, msg):
                print(f"⚠️ Warning: {msg}")
                
            def error(self, msg):
                print(f"❌ Error: {msg}")
        
        placeholder = MockPlaceholder()
        
        # Test the function
        safe_image_display(placeholder, dummy_image, "Test Image")
        print("✅ safe_image_display test passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_streamlit_version():
    """Test Streamlit version detection"""
    try:
        version = st.__version__
        print(f"📦 Streamlit version: {version}")
        
        # Test if use_container_width is supported
        import inspect
        sig = inspect.signature(st.image)
        
        if 'use_container_width' in sig.parameters:
            print("✅ use_container_width is supported")
        else:
            print("⚠️ use_container_width is NOT supported in this version")
            
        return True
        
    except Exception as e:
        print(f"❌ Version test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 CrowdGuard Pro Compatibility Testing")
    print("=" * 50)
    
    version_ok = test_streamlit_version()
    image_ok = test_image_display()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Version Detection: {'✅ PASS' if version_ok else '❌ FAIL'}")
    print(f"   Image Display: {'✅ PASS' if image_ok else '❌ FAIL'}")
    
    if version_ok and image_ok:
        print("\n🎉 All compatibility tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check compatibility issues above.")
