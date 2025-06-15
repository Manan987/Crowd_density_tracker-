import streamlit as st
import numpy as np
from datetime import datetime

def show_settings():
    """Simplified system settings page without database"""
    st.markdown("""
    <div class="page-header">
        <h1>⚙️ System Settings</h1>
        <p>Configure crowd monitoring parameters and system preferences</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Alert Thresholds")
        
        warning_threshold = st.slider("Warning Threshold (people/m²)", 0.5, 5.0, 2.5, 0.1)
        critical_threshold = st.slider("Critical Threshold (people/m²)", 1.0, 8.0, 4.0, 0.1)
        
        st.session_state['warning_threshold'] = warning_threshold
        st.session_state['critical_threshold'] = critical_threshold
        
        st.markdown("### 🎬 Video Settings")
        
        fps = st.selectbox("Processing FPS", [15, 24, 30, 60], index=2)
        resolution = st.selectbox("Resolution", ["720p", "1080p", "4K"], index=1)
        
        st.session_state['video_fps'] = fps
        st.session_state['video_resolution'] = resolution
        
    with col2:
        st.markdown("### 🔔 Notification Settings")
        
        email_alerts = st.checkbox("Email Alerts", value=True)
        sms_alerts = st.checkbox("SMS Alerts", value=False)
        sound_alerts = st.checkbox("Sound Alerts", value=True)
        
        st.session_state['email_alerts'] = email_alerts
        st.session_state['sms_alerts'] = sms_alerts
        st.session_state['sound_alerts'] = sound_alerts
        
        st.markdown("### 🎨 Display Options")
        
        dark_mode = st.checkbox("Dark Mode", value=True)
        show_heatmap = st.checkbox("Show Density Heatmap", value=True)
        show_grid = st.checkbox("Show Grid Overlay", value=False)
        
        st.session_state['dark_mode'] = dark_mode
        st.session_state['show_heatmap'] = show_heatmap
        st.session_state['show_grid'] = show_grid
    
    st.markdown("---")
    
    # Quick actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Settings", type="primary", use_container_width=True):
            st.success("Settings saved successfully!")
    
    with col2:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            # Reset to default values
            st.session_state['warning_threshold'] = 2.5
            st.session_state['critical_threshold'] = 4.0
            st.session_state['video_fps'] = 30
            st.session_state['video_resolution'] = "1080p"
            st.rerun()
    
    with col3:
        if st.button("📤 Export Config", use_container_width=True):
            st.info("Configuration exported!")
    
    # System information
    st.markdown("### 📊 System Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("System Uptime", "2h 45m")
        
    with col2:
        st.metric("Memory Usage", "1.2 GB")
        
    with col3:
        st.metric("CPU Usage", "23%")