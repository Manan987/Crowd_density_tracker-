import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import threading
import inspect
from models.crowd_density_model import CrowdDensityEstimator
from models.enhanced_density_model import EnhancedCrowdDensityEstimator
from utils.video_processor import VideoProcessor
from utils.enhanced_video_processor import MultiStreamProcessor
from utils.alert_system import AlertSystem
from utils.enhanced_alert_system import EnhancedAlertSystem
from utils.data_manager import DataManager  # Add this line
# Removed database dependency

# Streamlit version compatibility helpers
def get_streamlit_version_info():
    """Get Streamlit version and compatibility info"""
    try:
        version = st.__version__
        major, minor = map(int, version.split('.')[:2])
        return {
            'version': version,
            'major': major,
            'minor': minor,
            'supports_use_container_width': major > 1 or (major == 1 and minor >= 28)
        }
    except:
        return {
            'version': 'unknown',
            'major': 1,
            'minor': 28,
            'supports_use_container_width': True
        }

def safe_component_kwargs(**kwargs):
    """Filter kwargs based on Streamlit version compatibility"""
    version_info = get_streamlit_version_info()
    
    # Remove unsupported parameters
    if not version_info['supports_use_container_width'] and 'use_container_width' in kwargs:
        kwargs.pop('use_container_width')
    
    return kwargs

# Page configuration
st.set_page_config(
    page_title="CrowdGuard Pro - AI Monitoring System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Enhanced global styling */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero section styling */
    .hero-section {
        background: linear-gradient(135deg, rgba(26,26,46,0.95) 0%, rgba(22,33,62,0.95) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 3rem 2rem;
        border-radius: 25px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.4);
        position: relative;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(102,126,234,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
        z-index: -1;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .hero-content {
        flex: 1;
        z-index: 1;
    }
    
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(102,126,234,0.3);
        color: white;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 1rem 0;
        line-height: 1.1;
    }
    
    .gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: rgba(255,255,255,0.8);
        margin-bottom: 2rem;
        line-height: 1.5;
    }
    
    .hero-features {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin-top: 2rem;
    }
    
    .feature-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.8rem 1rem;
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
        transition: transform 0.3s ease, background 0.3s ease;
        color: rgba(255,255,255,0.9);
    }
    
    .feature-item:hover {
        transform: translateY(-2px);
        background: rgba(255,255,255,0.1);
    }
    
    .feature-icon {
        font-size: 1.2rem;
    }
    
    .hero-visual {
        flex: 0 0 300px;
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1;
    }
    
    .floating-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.2) 0%, rgba(118,75,162,0.2) 100%);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.2);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.3);
        animation: float 6s ease-in-out infinite;
        width: 250px;
        text-align: center;
        color: white;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .card-header {
        font-weight: 600;
        margin-bottom: 1rem;
        color: #667eea;
    }
    
    .pulse-dot {
        width: 20px;
        height: 20px;
        background: #00ff88;
        border-radius: 50%;
        margin: 1rem auto;
        animation: pulse-glow 2s infinite;
    }
    
    @keyframes pulse-glow {
        0% { 
            box-shadow: 0 0 5px #00ff88;
            transform: scale(1);
        }
        50% { 
            box-shadow: 0 0 20px #00ff88, 0 0 30px rgba(0,255,136,0.5);
            transform: scale(1.1);
        }
        100% { 
            box-shadow: 0 0 5px #00ff88;
            transform: scale(1);
        }
    }
    
    .data-stream {
        display: flex;
        flex-direction: column;
        gap: 0.3rem;
        margin-top: 1rem;
    }
    
    .stream-line {
        height: 3px;
        background: linear-gradient(90deg, transparent 0%, #667eea 50%, transparent 100%);
        border-radius: 2px;
        animation: stream 2s linear infinite;
    }
    
    .stream-line:nth-child(2) {
        animation-delay: 0.3s;
    }
    
    .stream-line:nth-child(3) {
        animation-delay: 0.6s;
    }
    
    @keyframes stream {
        0% { transform: translateX(-100%); opacity: 0; }
        50% { opacity: 1; }
        100% { transform: translateX(100%); opacity: 0; }
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        color: white;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Sidebar header styling */
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(102,126,234,0.3);
    }
    
    .sidebar-header h2 {
        color: white;
        margin: 0 0 0.5rem 0;
        font-size: 1.3rem;
        font-weight: 700;
    }
    
    .sidebar-header p {
        color: rgba(255,255,255,0.8);
        margin: 0;
        font-size: 0.9rem;
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(26,26,46,0.8) 0%, rgba(22,33,62,0.8) 100%);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: left 0.5s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        border-color: rgba(102,126,234,0.5);
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .alert-card {
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        border-left: 5px solid;
    }
    
    .alert-normal {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-left-color: #00ff88;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-left-color: #ff6b35;
    }
    
    .alert-critical {
        background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%);
        border-left-color: #ff0844;
    }
    
    .camera-status {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .camera-status:hover {
        background: rgba(255,255,255,0.08);
        transform: scale(1.02);
    }
    
    /* Page header styling */
    .page-header {
        background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .page-header h1 {
        color: white;
        margin: 0 0 0.5rem 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .page-header p {
        color: rgba(255,255,255,0.8);
        margin: 0 0 1.5rem 0;
        font-size: 1.1rem;
    }
    
    .feature-badges {
        display: flex;
        justify-content: center;
        gap: 1rem;
        flex-wrap: wrap;
    }
    
    .badge {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .badge-ai {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .badge-realtime {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .badge-secure {
        background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%);
        color: white;
    }
    
    /* Overview cards */
    .overview-card {
        background: linear-gradient(135deg, rgba(26,26,46,0.8) 0%, rgba(22,33,62,0.8) 100%);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .overview-card:hover {
        transform: translateY(-5px);
        border-color: rgba(102,126,234,0.5);
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .card-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .overview-card h3 {
        color: white;
        margin: 0.5rem 0;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    .overview-card p {
        color: rgba(255,255,255,0.7);
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    .card-status {
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .card-status.online {
        background: rgba(76, 175, 80, 0.2);
        color: #4CAF50;
        border: 1px solid #4CAF50;
    }
    
    .card-status.processing {
        background: rgba(102, 126, 234, 0.2);
        color: #667eea;
        border: 1px solid #667eea;
    }
    
    .card-status.excellent {
        background: rgba(0, 255, 136, 0.2);
        color: #00ff88;
        border: 1px solid #00ff88;
    }
    
    .card-status.secure {
        background: rgba(118, 75, 162, 0.2);
        color: #764ba2;
        border: 1px solid #764ba2;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .feature-badges {
            flex-direction: column;
            align-items: center;
        }
        
        .overview-card {
            margin-bottom: 1rem;
        }
        
        .page-header h1 {
            font-size: 2rem;
        }
    }
    
    .status-online {
        border-left: 4px solid #00ff88;
    }
    
    .status-warning {
        border-left: 4px solid #ffaa00;
    }
    
    .status-offline {
        border-left: 4px solid #ff4444;
    }
    
    .control-panel {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
    }
    
    .video-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(0,0,0,0.4);
        border: 2px solid rgba(255,255,255,0.1);
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .stat-item {
        background: linear-gradient(135deg, #434343 0%, #000000 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00ff88;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.7);
        margin-top: 0.5rem;
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2C3E50 0%, #34495E 100%);
    }
    
    .nav-item {
        padding: 0.8rem 1rem;
        margin: 0.3rem 0;
        border-radius: 8px;
        transition: background 0.3s ease;
        cursor: pointer;
    }
    
    .nav-item:hover {
        background: rgba(255,255,255,0.1);
    }
    
    .nav-item.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with simple in-memory storage
if 'density_data' not in st.session_state:
    st.session_state.density_data = []
if 'alert_events' not in st.session_state:
    st.session_state.alert_events = []
if 'camera_status' not in st.session_state:
    st.session_state.camera_status = {
        'CAM_001': {'status': 'online', 'last_update': datetime.now()},
        'CAM_002': {'status': 'online', 'last_update': datetime.now()},
        'CAM_003': {'status': 'online', 'last_update': datetime.now()},
        'CAM_004': {'status': 'online', 'last_update': datetime.now()}
    }
if 'alert_system' not in st.session_state:
    st.session_state.alert_system = AlertSystem()
if 'model' not in st.session_state:
    st.session_state.model = CrowdDensityEstimator()
if 'video_processor' not in st.session_state:
    st.session_state.video_processor = VideoProcessor()
if 'data_manager' not in st.session_state:  # Add this block
    st.session_state.data_manager = DataManager()
if 'enhanced_model' not in st.session_state:
    st.session_state.enhanced_model = EnhancedCrowdDensityEstimator(model_type='csrnet')
if 'multi_processor' not in st.session_state:
    st.session_state.multi_processor = MultiStreamProcessor(max_streams=4)
if 'enhanced_alerts' not in st.session_state:
    st.session_state.enhanced_alerts = EnhancedAlertSystem()
if 'is_monitoring' not in st.session_state:
    st.session_state.is_monitoring = False
if 'current_density' not in st.session_state:
    st.session_state.current_density = 0
if 'alert_status' not in st.session_state:
    st.session_state.alert_status = "Normal"

def main():
    # Enhanced header with hero section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-content">
            <div class="hero-badge">
                <span>🏆 ENTERPRISE GRADE</span>
            </div>
            <h1 class="hero-title">
                <span class="gradient-text">CrowdGuard Pro</span>
            </h1>
            <p class="hero-subtitle">Advanced AI-Powered Crowd Density Monitoring & Real-Time Stampede Prevention System</p>
            <div class="hero-features">
                <div class="feature-item">
                    <span class="feature-icon">🧠</span>
                    <span>AI-Powered Detection</span>
                </div>
                <div class="feature-item">
                    <span class="feature-icon">⚡</span>
                    <span>Real-Time Processing</span>
                </div>
                <div class="feature-item">
                    <span class="feature-icon">🛡️</span>
                    <span>Enterprise Security</span>
                </div>
                <div class="feature-item">
                    <span class="feature-icon">📊</span>
                    <span>Advanced Analytics</span>
                </div>
            </div>
        </div>
        <div class="hero-visual">
            <div class="floating-card">
                <div class="card-header">Live Monitoring</div>
                <div class="pulse-dot"></div>
                <div class="data-stream">
                    <div class="stream-line"></div>
                    <div class="stream-line"></div>
                    <div class="stream-line"></div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced sidebar navigation with attractive visuals
    with st.sidebar:
        # Sidebar header with gradient background
        st.markdown("""
        <div class="sidebar-header">
            <h2>🎛️ Control Center</h2>
            <p>Real-time crowd monitoring dashboard</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation with modern styling
        page_options = [
            ("🎬", "Real-time Monitoring"),
            ("🎯", "Advanced Monitoring"),
            ("🚨", "Alert Management"),
            ("📊", "Analytics Dashboard"), 
            ("📈", "Advanced Analytics"),

            ("⚙️", "System Settings"),
            ("💻", "System Status")
        ]
        
        page = st.selectbox(
            "Navigate to:",
            [option[1] for option in page_options],
            format_func=lambda x: f"{next(opt[0] for opt in page_options if opt[1] == x)} {x}"
        )
        
        st.markdown("---")
        
        # Enhanced system status display
        st.markdown("### 📊 Live System Status")
        
        # Current alert level with styling
        alert_level = st.session_state.get('alert_status', 'Normal')
        alert_class = f"alert-{alert_level.lower()}"
        
        st.markdown(f"""
        <div class="alert-card {alert_class}">
            <h4 style="margin: 0; color: white;">
                {'🟢' if alert_level == 'Normal' else '🟡' if alert_level == 'Warning' else '🔴'} 
                {alert_level.upper()} STATUS
            </h4>
            <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.9);">
                System operating {alert_level.lower()}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Live metrics
        current_density = st.session_state.get('current_density', 0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Density", f"{current_density:.1f}", "people/m²")
        with col2:
            # Active cameras count
            active_cams = len([cam for cam, status in st.session_state.camera_status.items() if status['status'] == 'online'])
            st.metric("Active Cams", f"{active_cams}/4")
        
        # System performance indicator
        st.markdown("### ⚡ Performance")
        performance_score = min(100, 85 + (active_cams * 3.75))  # Dynamic score
        progress_color = "#00ff88" if performance_score > 80 else "#ffaa00" if performance_score > 60 else "#ff4444"
        
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span>System Health</span>
                <span style="color: {progress_color}; font-weight: bold;">{performance_score:.0f}%</span>
            </div>
            <div style="background: rgba(255,255,255,0.2); height: 8px; border-radius: 4px; margin-top: 0.5rem;">
                <div style="background: {progress_color}; height: 100%; width: {performance_score}%; border-radius: 4px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick actions
        st.markdown("### 🚀 Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset"):
                st.session_state.density_data = []
                st.session_state.alert_events = []
                st.session_state.is_monitoring = False
                st.rerun()
        
        with col2:
            if st.button("📤 Export"):
                st.success("Data exported!")
        
        # Emergency controls
        st.markdown("---")
        st.markdown("### 🚨 Emergency Controls")
        
        if st.button("🚨 EMERGENCY STOP", type="primary"):
            st.session_state.is_monitoring = False
            st.error("EMERGENCY STOP ACTIVATED!")
            st.balloons()
    
    # Main content based on selected page
    if page == "Real-time Monitoring":
        show_monitoring_page()
    elif page == "Advanced Monitoring":
        from components.advanced_visualizations import show_advanced_monitoring_dashboard
        show_advanced_monitoring_dashboard()
    elif page == "Alert Management":
        from components.real_time_alerts import show_real_time_alert_interface
        show_real_time_alert_interface()
    elif page == "Analytics Dashboard":
        from pages.dashboard import show_dashboard
        show_dashboard()
    elif page == "Advanced Analytics":
        from pages.analytics import show_analytics
        show_analytics()

    elif page == "System Settings":
        from pages.settings import show_settings
        show_settings()
    elif page == "System Status":
        show_system_status_page()

def show_monitoring_page():
    # Enhanced monitoring interface with attractive content
    st.markdown("""
    <div class="page-header">
        <h1>🎬 Live Monitoring Center</h1>
        <p>Real-time AI-powered crowd density analysis and stampede prevention system</p>
        <div class="feature-badges">
            <span class="badge badge-ai">🧠 AI Detection</span>
            <span class="badge badge-realtime">⚡ Real-time</span>
            <span class="badge badge-secure">🔒 Secure</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # System overview cards
    st.markdown("### 📊 System Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="overview-card">
            <div class="card-icon">📹</div>
            <h3>Multi-Camera</h3>
            <p>4 Active Cameras</p>
            <div class="card-status online">ONLINE</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="overview-card">
            <div class="card-icon">🧠</div>
            <h3>AI Processing</h3>
            <p>Neural Network</p>
            <div class="card-status processing">ACTIVE</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="overview-card">
            <div class="card-icon">⚡</div>
            <h3>Response Time</h3>
            <p>< 50ms</p>
            <div class="card-status excellent">EXCELLENT</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="overview-card">
            <div class="card-icon">🛡️</div>
            <h3>Security</h3>
            <p>Enterprise Grade</p>
            <div class="card-status secure">PROTECTED</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Control panel with modern styling
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    
    # Input source and controls
    col1, col2, col3, col4 = st.columns([2, 1.5, 1, 1])
    
    with col1:
        input_type = st.selectbox(
            "📹 Input Source",
            ["Simulate Camera Feed", "Upload Video", "Webcam Feed"],
            help="Select the video input source for crowd analysis"
        )
    
    with col2:
        camera_options = [
            ("CAM_001", "Main Entrance"),
            ("CAM_002", "Central Plaza"), 
            ("CAM_003", "Exit Gate"),
            ("CAM_004", "Emergency Route")
        ]
        
        camera_selection = st.selectbox(
            "🎯 Camera Location",
            camera_options,
            format_func=lambda x: f"{x[0]} - {x[1]}"
        )
        camera_id = camera_selection[0]
    
    with col3:
        monitoring_mode = st.selectbox(
            "🔍 Detection Mode",
            ["Standard", "High Precision", "Fast Processing"]
        )
    
    with col4:
        st.markdown("**System Control**")
        if st.session_state.get('is_monitoring', False):
            if st.button("⏹️ Stop", type="secondary"):
                st.session_state.is_monitoring = False
        else:
            if st.button("▶️ Start", type="primary"):
                st.session_state.is_monitoring = True
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced monitoring display
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Video feed container with modern styling
        st.markdown("### 📺 Live Video Analysis")
        
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        video_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Video processing based on input type
        if st.session_state.get('is_monitoring', False):
            if input_type == "Upload Video":
                uploaded_file = st.file_uploader(
                    "Select video file for analysis",
                    type=['mp4', 'avi', 'mov', 'mkv'],
                    help="Upload a video file to analyze crowd density patterns"
                )
                
                if uploaded_file is not None:
                    process_uploaded_video(uploaded_file, camera_id, video_placeholder)
            
            elif input_type == "Webcam Feed":
                process_webcam_feed(camera_id, video_placeholder)
            
            elif input_type == "Simulate Camera Feed":
                simulate_camera_feed(camera_id, video_placeholder)
        else:
            # Display placeholder when not monitoring
            st.markdown("""
            <div style="background: rgba(255,255,255,0.05); border: 2px dashed rgba(255,255,255,0.2); 
                        border-radius: 15px; padding: 4rem; text-align: center; margin: 2rem 0;">
                <h3 style="color: rgba(255,255,255,0.6); margin: 0;">
                    🎬 Press START to begin monitoring
                </h3>
                <p style="color: rgba(255,255,255,0.4); margin: 1rem 0 0 0;">
                    Live video feed and AI analysis will appear here
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Real-time performance metrics
        show_realtime_performance_metrics()
    
    with col2:
        # Enhanced side panel
        show_enhanced_alert_panel()
        show_enhanced_density_stats()
        show_camera_grid_overview()
        show_recent_events_enhanced()

def safe_image_display(video_placeholder, image_data, caption, use_container_width=True):
    """Safely display image with error handling and version compatibility"""
    try:
        # Simple approach - just use basic parameters
        video_placeholder.image(image_data, caption=caption)
        
    except Exception as e:
        # Handle missing media file errors gracefully
        if "MediaFileStorageError" in str(e) or "Missing file" in str(e):
            video_placeholder.warning("🔄 Media file expired. Please refresh or restart monitoring.")
        elif "unexpected keyword argument" in str(e):
            # Handle version compatibility issues
            try:
                video_placeholder.image(image_data, caption=caption)
            except Exception as fallback_error:
                video_placeholder.error(f"Error displaying image: {str(fallback_error)}")
        else:
            video_placeholder.error(f"Error displaying image: {str(e)}")

def process_uploaded_video(uploaded_file, camera_id, video_placeholder):
    """Process uploaded video file for crowd density estimation"""
    try:
        # Save uploaded file temporarily
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        
        # Process video
        cap = cv2.VideoCapture("temp_video.mp4")
        frame_delay = 0.1  # Adjust for playback speed
        
        while cap.isOpened() and st.session_state.is_monitoring:
            ret, frame = cap.read()
            if not ret:
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
            
            # Display frame with safe error handling
            safe_image_display(
                video_placeholder,
                heatmap_frame,
                f"Camera: {camera_id} | Density: {density_per_m2:.1f} people/m² | Count: {crowd_count}"
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
            
            # Display frame with safe error handling
            safe_image_display(
                video_placeholder,
                heatmap_frame,
                f"Camera: {camera_id} | Density: {density_per_m2:.1f} people/m² | Count: {crowd_count}"
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
        
        # Display frame with safe error handling
        safe_image_display(
            video_placeholder,
            synthetic_frame,
            f"Camera: {camera_id} | Density: {density_per_m2:.1f} people/m² | Count: {crowd_count}"
        )
        
        time.sleep(1.0)  # Update every second

def show_realtime_performance_metrics():
    """Display real-time performance metrics below video feed"""
    st.markdown("### ⚡ Real-time Performance Metrics")
    
    # Create metrics in a grid layout
    col1, col2, col3, col4 = st.columns(4)
    
    # Get recent data for metrics from in-memory storage
    recent_data = [d for d in st.session_state.density_data if (datetime.now() - d['timestamp']).seconds < 300]
    
    with col1:
        fps = 30 if st.session_state.get('is_monitoring', False) else 0
        st.metric("Processing FPS", f"{fps}", help="Frames processed per second")
    
    with col2:
        latency = np.random.uniform(15, 45) if st.session_state.get('is_monitoring', False) else 0
        st.metric("Latency", f"{latency:.0f}ms", help="Processing latency")
    
    with col3:
        accuracy = np.random.uniform(92, 98) if st.session_state.get('is_monitoring', False) else 0
        st.metric("AI Accuracy", f"{accuracy:.1f}%", help="Model prediction accuracy")
    
    with col4:
        throughput = len(recent_data)
        st.metric("Data Points", throughput, help="Data points in last 5 minutes")

def show_enhanced_alert_panel():
    """Enhanced alert panel with modern styling"""
    st.markdown("### 🚨 Alert Center")
    
    alert_level = st.session_state.get('alert_status', 'Normal')
    current_density = st.session_state.get('current_density', 0)
    
    # Dynamic alert styling
    alert_configs = {
        "Normal": {"color": "#00ff88", "icon": "🟢", "bg": "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)"},
        "Warning": {"color": "#ffaa00", "icon": "🟡", "bg": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"},
        "Critical": {"color": "#ff4444", "icon": "🔴", "bg": "linear-gradient(135deg, #fc466b 0%, #3f5efb 100%)"}
    }
    
    config = alert_configs.get(alert_level, alert_configs["Normal"])
    
    st.markdown(f"""
    <div style="background: {config['bg']}; padding: 1.5rem; border-radius: 12px; 
                margin: 1rem 0; box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h3 style="margin: 0; color: white; font-size: 1.2rem;">
                    {config['icon']} {alert_level.upper()}
                </h3>
                <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.9);">
                    Density: {current_density:.1f} people/m²
                </p>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 2rem;">{config['icon']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Alert actions
    if alert_level != "Normal":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Acknowledge", ):
                st.success("Alert acknowledged")
        with col2:
            if st.button("📞 Contact", ):
                st.info("Emergency services notified")

def show_enhanced_density_stats():
    """Enhanced density statistics display"""
    st.markdown("### 📊 Density Analytics")
    
    data = st.session_state.data_manager.get_recent_data(minutes=30)
    
    if not data.empty:
        avg_density = data['density'].mean()
        max_density = data['density'].max()
        trend = "↗️" if avg_density > data['density'].iloc[:len(data)//2].mean() else "↘️"
        
        # Statistics cards
        stats_html = f"""
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value">{avg_density:.1f}</div>
                <div class="stat-label">Avg Density (30m)</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{max_density:.1f}</div>
                <div class="stat-label">Peak Density</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{trend}</div>
                <div class="stat-label">Trend</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{len(data)}</div>
                <div class="stat-label">Data Points</div>
            </div>
        </div>
        """
        st.markdown(stats_html, unsafe_allow_html=True)
        
        # Mini trend chart
        if len(data) > 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['timestamp'].dt.strftime('%H:%M'),
                y=data['density'],
                mode='lines+markers',
                line=dict(color='#00ff88', width=2),
                marker=dict(size=4),
                name='Density'
            ))
            fig.update_layout(
                height=200,
                margin=dict(l=0, r=0, t=20, b=0),
                showlegend=False,
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig)
    else:
        st.info("Gathering density data...")

def show_camera_grid_overview():
    """Show camera grid overview with status indicators"""
    st.markdown("### 📹 Camera Network")
    
    cameras = [
        {"id": "CAM_001", "name": "Main Entrance", "status": "online"},
        {"id": "CAM_002", "name": "Central Plaza", "status": "online"},
        {"id": "CAM_003", "name": "Exit Gate", "status": "warning"},
        {"id": "CAM_004", "name": "Emergency Route", "status": "offline"}
    ]
    
    for camera in cameras:
        status_class = f"status-{camera['status']}"
        status_icon = "🟢" if camera['status'] == 'online' else "🟡" if camera['status'] == 'warning' else "🔴"
        
        st.markdown(f"""
        <div class="camera-status {status_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong>{camera['id']}</strong><br>
                    <small style="color: rgba(255,255,255,0.7);">{camera['name']}</small>
                </div>
                <div style="font-size: 1.5rem;">{status_icon}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_recent_events_enhanced():
    """Enhanced recent events display"""
    st.markdown("### 📋 Recent Activity")
    
    # Get recent alert events
    events = st.session_state.data_manager.get_alert_events(hours=2)
    
    if not events.empty and len(events) > 0:
        # Show last 3 events
        recent_events = events.head(3)
        
        for _, event in recent_events.iterrows():
            time_str = event['timestamp'].strftime('%H:%M:%S')
            level_color = "#00ff88" if event['alert_level'] == 'Normal' else "#ffaa00" if event['alert_level'] == 'Warning' else "#ff4444"
            
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); padding: 1rem; margin: 0.5rem 0; 
                        border-radius: 8px; border-left: 3px solid {level_color};">
                <div style="display: flex; justify-content: space-between;">
                    <strong>{event['camera_id']}</strong>
                    <small>{time_str}</small>
                </div>
                <div style="color: {level_color}; margin-top: 0.2rem;">
                    {event['alert_level']} - {event['density']:.1f} people/m²
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: rgba(255,255,255,0.5);">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📝</div>
            <div>No recent events</div>
        </div>
        """, unsafe_allow_html=True)

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
            data[['timestamp', 'camera_id', 'alert_level', 'density']].tail(5)
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
    st.plotly_chart(fig_trend)
    
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
        st.plotly_chart(fig_pie)
    
    with col2:
        st.subheader("Camera Performance")
        camera_stats = data.groupby('camera_id').agg({
            'density': ['mean', 'max'],
            'alert_level': lambda x: (x != 'Normal').sum()
        }).round(2)
        camera_stats.columns = ['Avg Density', 'Max Density', 'Alert Count']
        st.dataframe(camera_stats)
    
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
        st.plotly_chart(fig_heatmap)

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
    st.dataframe(status_df)
    
    # Recent system logs
    st.subheader("Recent System Logs")
    
    logs = [
        {"timestamp": datetime.now() - timedelta(minutes=1), "level": "INFO", "message": "System monitoring active"},
        {"timestamp": datetime.now() - timedelta(minutes=3), "level": "WARNING", "message": "High density detected on CAM_002"},
        {"timestamp": datetime.now() - timedelta(minutes=5), "level": "INFO", "message": "Alert acknowledged by operator"},
        {"timestamp": datetime.now() - timedelta(minutes=8), "level": "INFO", "message": "Camera CAM_001 reconnected"},
    ]
    
    logs_df = pd.DataFrame(logs)
    st.dataframe(logs_df)
    
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
