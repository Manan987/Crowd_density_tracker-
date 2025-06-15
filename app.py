import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import threading
from models.crowd_density_model import CrowdDensityEstimator
from utils.video_processor import VideoProcessor
from utils.alert_system import AlertSystem
from utils.data_manager import DataManager

# Page configuration
st.set_page_config(
    page_title="Crowd Density Monitoring System",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = DataManager()
if 'alert_system' not in st.session_state:
    st.session_state.alert_system = AlertSystem()
if 'model' not in st.session_state:
    st.session_state.model = CrowdDensityEstimator()
if 'video_processor' not in st.session_state:
    st.session_state.video_processor = VideoProcessor()
if 'is_monitoring' not in st.session_state:
    st.session_state.is_monitoring = False
if 'current_density' not in st.session_state:
    st.session_state.current_density = 0
if 'alert_status' not in st.session_state:
    st.session_state.alert_status = "Normal"

def main():
    st.title("🚨 Crowd Density Monitoring System")
    st.markdown("---")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Select Page",
            ["Real-time Monitoring", "Analytics", "Settings", "System Status"]
        )
        
        st.markdown("---")
        st.header("System Status")
        
        # Display current status
        status_color = {
            "Normal": "🟢",
            "Warning": "🟡", 
            "Critical": "🔴"
        }
        
        st.metric(
            "Current Alert Level",
            f"{status_color.get(st.session_state.alert_status, '⚪')} {st.session_state.alert_status}",
            delta=None
        )
        
        st.metric(
            "Current Density",
            f"{st.session_state.current_density:.1f} people/m²"
        )
        
        # System controls
        st.markdown("---")
        st.header("System Controls")
        
        if st.button("🔄 Reset System"):
            st.session_state.data_manager.clear_data()
            st.session_state.is_monitoring = False
            st.rerun()
    
    # Main content based on selected page
    if page == "Real-time Monitoring":
        show_monitoring_page()
    elif page == "Analytics":
        show_analytics_page()
    elif page == "Settings":
        show_settings_page()
    elif page == "System Status":
        show_system_status_page()

def show_monitoring_page():
    st.header("📹 Real-time Crowd Density Monitoring")
    
    # Input source selection
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        input_type = st.selectbox(
            "Select Input Source",
            ["Upload Video", "Webcam", "Simulate Camera Feed"]
        )
    
    with col2:
        camera_id = st.selectbox(
            "Camera ID",
            ["CAM_001", "CAM_002", "CAM_003", "CAM_004"]
        )
    
    with col3:
        if st.button("🎬 Start Monitoring"):
            st.session_state.is_monitoring = True
        if st.button("⏹️ Stop Monitoring"):
            st.session_state.is_monitoring = False
    
    st.markdown("---")
    
    # Main monitoring display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Feed & Density Heatmap")
        video_placeholder = st.empty()
        
        if input_type == "Upload Video":
            uploaded_file = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'avi', 'mov', 'mkv']
            )
            
            if uploaded_file is not None and st.session_state.is_monitoring:
                process_uploaded_video(uploaded_file, camera_id, video_placeholder)
        
        elif input_type == "Webcam":
            if st.session_state.is_monitoring:
                process_webcam_feed(camera_id, video_placeholder)
        
        elif input_type == "Simulate Camera Feed":
            if st.session_state.is_monitoring:
                simulate_camera_feed(camera_id, video_placeholder)
    
    with col2:
        st.subheader("Alert Panel")
        show_alert_panel()
        
        st.subheader("Density Statistics")
        show_density_stats()
        
        st.subheader("Recent Events")
        show_recent_events()

def process_uploaded_video(uploaded_file, camera_id, video_placeholder):
    """Process uploaded video file for crowd density estimation"""
    try:
        # Save uploaded file temporarily
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        
        cap = cv2.VideoCapture("temp_video.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / fps if fps > 0 else 0.033  # Default to ~30fps
        
        while cap.isOpened() and st.session_state.is_monitoring:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                continue
            
            # Process frame for crowd density
            density_map, crowd_count = st.session_state.model.estimate_density(frame)
            
            # Calculate density per square meter (assuming 1 pixel = 0.1m for demo)
            area_m2 = (frame.shape[0] * frame.shape[1]) * (0.1 * 0.1)
            density_per_m2 = crowd_count / area_m2 if area_m2 > 0 else 0
            
            # Update global state
            st.session_state.current_density = density_per_m2
            
            # Generate heatmap overlay
            heatmap_frame = st.session_state.video_processor.create_heatmap_overlay(
                frame, density_map
            )
            
            # Check alerts
            alert_level = st.session_state.alert_system.check_alert_level(density_per_m2)
            st.session_state.alert_status = alert_level
            
            # Store data
            st.session_state.data_manager.add_data_point(
                camera_id, density_per_m2, crowd_count, alert_level
            )
            
            # Display frame
            video_placeholder.image(
                heatmap_frame,
                caption=f"Camera: {camera_id} | Density: {density_per_m2:.1f} people/m² | Count: {crowd_count}",
                use_column_width=True
            )
            
            time.sleep(frame_delay)
        
        cap.release()
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")

def process_webcam_feed(camera_id, video_placeholder):
    """Process webcam feed for crowd density estimation"""
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Unable to access webcam. Please check your camera permissions.")
            return
        
        while cap.isOpened() and st.session_state.is_monitoring:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from webcam")
                break
            
            # Process frame for crowd density
            density_map, crowd_count = st.session_state.model.estimate_density(frame)
            
            # Calculate density per square meter
            area_m2 = (frame.shape[0] * frame.shape[1]) * (0.1 * 0.1)
            density_per_m2 = crowd_count / area_m2 if area_m2 > 0 else 0
            
            # Update global state
            st.session_state.current_density = density_per_m2
            
            # Generate heatmap overlay
            heatmap_frame = st.session_state.video_processor.create_heatmap_overlay(
                frame, density_map
            )
            
            # Check alerts
            alert_level = st.session_state.alert_system.check_alert_level(density_per_m2)
            st.session_state.alert_status = alert_level
            
            # Store data
            st.session_state.data_manager.add_data_point(
                camera_id, density_per_m2, crowd_count, alert_level
            )
            
            # Display frame
            video_placeholder.image(
                heatmap_frame,
                caption=f"Camera: {camera_id} | Density: {density_per_m2:.1f} people/m² | Count: {crowd_count}",
                use_column_width=True
            )
            
            time.sleep(0.033)  # ~30fps
        
        cap.release()
        
    except Exception as e:
        st.error(f"Error processing webcam: {str(e)}")

def simulate_camera_feed(camera_id, video_placeholder):
    """Simulate camera feed with synthetic crowd density data"""
    import random
    
    while st.session_state.is_monitoring:
        # Generate synthetic density data
        base_density = random.uniform(0.5, 3.0)
        time_factor = np.sin(time.time() * 0.1) * 0.5 + 0.5  # Oscillating pattern
        density_per_m2 = base_density * (1 + time_factor)
        crowd_count = int(density_per_m2 * 100)  # Simulate area of 100 m²
        
        # Update global state
        st.session_state.current_density = density_per_m2
        
        # Create synthetic heatmap visualization
        synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some random "hot spots" to simulate crowd areas
        for _ in range(int(density_per_m2 * 3)):
            x = random.randint(50, 590)
            y = random.randint(50, 430)
            intensity = int(255 * min(density_per_m2 / 4.0, 1.0))
            cv2.circle(synthetic_frame, (x, y), random.randint(20, 60), 
                      (0, intensity, 255 - intensity), -1)
        
        # Add text overlay
        cv2.putText(synthetic_frame, f"SIMULATED FEED - {camera_id}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(synthetic_frame, f"Density: {density_per_m2:.1f} people/m²", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Check alerts
        alert_level = st.session_state.alert_system.check_alert_level(density_per_m2)
        st.session_state.alert_status = alert_level
        
        # Store data
        st.session_state.data_manager.add_data_point(
            camera_id, density_per_m2, crowd_count, alert_level
        )
        
        # Display frame
        video_placeholder.image(
            synthetic_frame,
            caption=f"Camera: {camera_id} | Density: {density_per_m2:.1f} people/m² | Count: {crowd_count}",
            use_column_width=True
        )
        
        time.sleep(1.0)  # Update every second

def show_alert_panel():
    """Display current alert status and controls"""
    alert_colors = {
        "Normal": "#28a745",
        "Warning": "#ffc107", 
        "Critical": "#dc3545"
    }
    
    current_color = alert_colors.get(st.session_state.alert_status, "#6c757d")
    
    st.markdown(f"""
    <div style="padding: 1rem; border-radius: 0.5rem; border: 2px solid {current_color}; 
                background-color: {current_color}20; margin-bottom: 1rem;">
        <h3 style="color: {current_color}; margin: 0;">
            {st.session_state.alert_status.upper()} ALERT
        </h3>
        <p style="margin: 0.5rem 0 0 0;">
            Current density: {st.session_state.current_density:.1f} people/m²
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.alert_status != "Normal":
        if st.button("🔕 Acknowledge Alert"):
            st.success("Alert acknowledged by operator")
        
        if st.button("📞 Emergency Contact"):
            st.info("Emergency services contacted automatically")

def show_density_stats():
    """Display density statistics"""
    data = st.session_state.data_manager.get_recent_data(minutes=10)
    
    if not data.empty:
        avg_density = data['density'].mean()
        max_density = data['density'].max()
        min_density = data['density'].min()
        
        st.metric("Average Density (10min)", f"{avg_density:.1f}", 
                 delta=f"{avg_density - st.session_state.current_density:.1f}")
        st.metric("Max Density (10min)", f"{max_density:.1f}")
        st.metric("Min Density (10min)", f"{min_density:.1f}")
    else:
        st.info("No data available yet")

def show_recent_events():
    """Display recent alert events"""
    data = st.session_state.data_manager.get_alert_events(hours=1)
    
    if not data.empty:
        st.dataframe(
            data[['timestamp', 'camera_id', 'alert_level', 'density']].tail(5),
            use_container_width=True
        )
    else:
        st.info("No recent events")

def show_analytics_page():
    """Analytics and historical data page"""
    st.header("📊 Analytics & Historical Data")
    
    # Time range selector
    col1, col2 = st.columns(2)
    with col1:
        time_range = st.selectbox(
            "Select Time Range",
            ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last Week"]
        )
    
    with col2:
        camera_filter = st.multiselect(
            "Filter by Camera",
            ["CAM_001", "CAM_002", "CAM_003", "CAM_004"],
            default=["CAM_001", "CAM_002", "CAM_003", "CAM_004"]
        )
    
    # Get data based on selection
    hours_map = {
        "Last Hour": 1,
        "Last 6 Hours": 6, 
        "Last 24 Hours": 24,
        "Last Week": 168
    }
    
    data = st.session_state.data_manager.get_recent_data(
        hours=hours_map[time_range],
        cameras=camera_filter
    )
    
    if data.empty:
        st.warning("No data available for the selected time range and cameras.")
        return
    
    # Density trend chart
    st.subheader("Density Trends")
    fig_trend = px.line(
        data, 
        x='timestamp', 
        y='density', 
        color='camera_id',
        title="Crowd Density Over Time",
        labels={'density': 'Density (people/m²)', 'timestamp': 'Time'}
    )
    fig_trend.add_hline(y=2.0, line_dash="dash", line_color="orange", 
                        annotation_text="Warning Threshold")
    fig_trend.add_hline(y=4.0, line_dash="dash", line_color="red", 
                        annotation_text="Critical Threshold")
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Alert frequency analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Alert Distribution")
        alert_counts = data['alert_level'].value_counts()
        fig_pie = px.pie(
            values=alert_counts.values,
            names=alert_counts.index,
            title="Alert Level Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Camera Performance")
        camera_stats = data.groupby('camera_id').agg({
            'density': ['mean', 'max'],
            'alert_level': lambda x: (x != 'Normal').sum()
        }).round(2)
        camera_stats.columns = ['Avg Density', 'Max Density', 'Alert Count']
        st.dataframe(camera_stats, use_container_width=True)
    
    # Heatmap by hour
    st.subheader("Density Heatmap by Hour")
    data['hour'] = data['timestamp'].dt.hour
    hourly_data = data.groupby(['hour', 'camera_id'])['density'].mean().reset_index()
    
    if not hourly_data.empty:
        fig_heatmap = px.density_heatmap(
            hourly_data,
            x='hour',
            y='camera_id', 
            z='density',
            title="Average Density by Hour and Camera"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

def show_settings_page():
    """System settings and configuration page"""
    st.header("⚙️ System Settings")
    
    # Alert thresholds
    st.subheader("Alert Thresholds")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        warning_threshold = st.number_input(
            "Warning Threshold (people/m²)",
            min_value=0.1,
            max_value=10.0,
            value=st.session_state.alert_system.warning_threshold,
            step=0.1
        )
    
    with col2:
        critical_threshold = st.number_input(
            "Critical Threshold (people/m²)",
            min_value=0.1,
            max_value=10.0,
            value=st.session_state.alert_system.critical_threshold,
            step=0.1
        )
    
    with col3:
        alert_duration = st.number_input(
            "Alert Duration (seconds)",
            min_value=1,
            max_value=300,
            value=st.session_state.alert_system.alert_duration,
            step=1
        )
    
    if st.button("Update Thresholds"):
        st.session_state.alert_system.update_thresholds(
            warning_threshold, critical_threshold, alert_duration
        )
        st.success("Thresholds updated successfully!")
    
    # Camera settings
    st.subheader("Camera Configuration")
    
    camera_locations = {
        "CAM_001": st.text_input("Camera 1 Location", value="Main Entrance"),
        "CAM_002": st.text_input("Camera 2 Location", value="Event Stage"),
        "CAM_003": st.text_input("Camera 3 Location", value="Food Court"),
        "CAM_004": st.text_input("Camera 4 Location", value="Exit Gate")
    }
    
    if st.button("Update Camera Locations"):
        st.success("Camera locations updated!")
    
    # Notification settings
    st.subheader("Notification Settings")
    
    email_notifications = st.checkbox("Enable Email Notifications", value=True)
    sms_notifications = st.checkbox("Enable SMS Notifications", value=False)
    
    if email_notifications:
        email_addresses = st.text_area(
            "Email Addresses (one per line)",
            value="security@venue.com\nmanager@venue.com"
        )
    
    if sms_notifications:
        phone_numbers = st.text_area(
            "Phone Numbers (one per line)",
            value="+1234567890"
        )
    
    # Data retention
    st.subheader("Data Management")
    
    retention_days = st.number_input(
        "Data Retention (days)",
        min_value=1,
        max_value=365,
        value=30,
        step=1
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export Historical Data"):
            csv_data = st.session_state.data_manager.export_data()
            st.download_button(
                "Download CSV",
                csv_data,
                file_name=f"crowd_density_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Clear All Data"):
            if st.session_state.get('confirm_clear'):
                st.session_state.data_manager.clear_data()
                st.success("All data cleared!")
                st.session_state.confirm_clear = False
            else:
                st.session_state.confirm_clear = True
                st.warning("Click again to confirm data deletion")

def show_system_status_page():
    """System status and health monitoring page"""
    st.header("🔧 System Status")
    
    # System health metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("System Uptime", "99.8%", delta="0.1%")
    
    with col2:
        st.metric("Active Cameras", "4/4", delta="0")
    
    with col3:
        st.metric("Processing Latency", "120ms", delta="-5ms")
    
    with col4:
        st.metric("Alert Response Time", "2.3s", delta="0.1s")
    
    # Component status
    st.subheader("Component Status")
    
    components = [
        {"name": "ML Model", "status": "✅ Online", "last_update": "2 minutes ago"},
        {"name": "Video Processor", "status": "✅ Online", "last_update": "1 minute ago"},
        {"name": "Alert System", "status": "✅ Online", "last_update": "30 seconds ago"},
        {"name": "Data Manager", "status": "✅ Online", "last_update": "15 seconds ago"},
    ]
    
    status_df = pd.DataFrame(components)
    st.dataframe(status_df, use_container_width=True)
    
    # Recent system logs
    st.subheader("Recent System Logs")
    
    logs = [
        {"timestamp": datetime.now() - timedelta(minutes=1), "level": "INFO", "message": "System monitoring active"},
        {"timestamp": datetime.now() - timedelta(minutes=3), "level": "WARNING", "message": "High density detected on CAM_002"},
        {"timestamp": datetime.now() - timedelta(minutes=5), "level": "INFO", "message": "Alert acknowledged by operator"},
        {"timestamp": datetime.now() - timedelta(minutes=8), "level": "INFO", "message": "Camera CAM_001 reconnected"},
    ]
    
    logs_df = pd.DataFrame(logs)
    st.dataframe(logs_df, use_container_width=True)
    
    # System diagnostics
    st.subheader("System Diagnostics")
    
    if st.button("Run System Diagnostics"):
        with st.spinner("Running diagnostics..."):
            time.sleep(2)
            st.success("✅ All systems operational")
            st.info("📊 Performance metrics within normal range")
            st.info("🔒 Security protocols active")
            st.info("💾 Data backup completed successfully")

if __name__ == "__main__":
    main()
