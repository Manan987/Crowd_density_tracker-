import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

def show_dashboard():
    """Simplified dashboard without database dependencies"""
    st.markdown("""
    <div class="page-header">
        <h1>📊 Analytics Dashboard</h1>
        <p>Real-time crowd monitoring analytics and insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate sample data for demonstration
    hours = np.arange(24)
    density_data = np.random.uniform(0.5, 4.0, 24) + np.sin(hours * np.pi / 12) * 1.5 + 2
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_density = np.mean(density_data)
        st.metric("Avg Density", f"{avg_density:.1f} p/m²", "↗ +0.3")
    
    with col2:
        max_density = np.max(density_data)
        st.metric("Peak Density", f"{max_density:.1f} p/m²", "↗ +0.8")
    
    with col3:
        alerts_today = np.random.randint(3, 12)
        st.metric("Alerts Today", alerts_today, "↘ -2")
    
    with col4:
        uptime = "99.8%"
        st.metric("System Uptime", uptime, "→ 0%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Density Trends")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours,
            y=density_data,
            mode='lines+markers',
            name='Density',
            line=dict(color='#667eea', width=3),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Density (people/m²)",
            height=300,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Alert Distribution")
        
        alert_types = ['Normal', 'Warning', 'Critical']
        alert_counts = [70, 25, 5]
        colors = ['#00ff88', '#ffaa00', '#ff4444']
        
        fig = go.Figure(data=[go.Pie(
            labels=alert_types,
            values=alert_counts,
            hole=0.4,
            marker_colors=colors
        )])
        
        fig.update_layout(
            height=300,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Camera overview
    st.markdown("### 📹 Camera Status Overview")
    
    camera_data = {
        'Camera': ['CAM_001', 'CAM_002', 'CAM_003', 'CAM_004'],
        'Status': ['Online', 'Online', 'Online', 'Maintenance'],
        'Current Density': [2.3, 1.8, 3.1, 0.0],
        'Last Alert': ['2 hrs ago', 'Never', '45 min ago', 'N/A']
    }
    
    df = pd.DataFrame(camera_data)
    
    # Style the dataframe
    def color_status(val):
        if val == 'Online':
            return 'background-color: rgba(0, 255, 136, 0.2); color: #00ff88'
        elif val == 'Maintenance':
            return 'background-color: rgba(255, 170, 0, 0.2); color: #ffaa00'
        return ''
    
    styled_df = df.style.applymap(color_status, subset=['Status'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Recent activity
    st.markdown("### 📋 Recent Activity")
    
    activity_data = [
        {"Time": "14:23", "Event": "Warning threshold exceeded", "Camera": "CAM_003", "Severity": "Warning"},
        {"Time": "13:45", "Event": "Normal operation resumed", "Camera": "CAM_001", "Severity": "Info"},
        {"Time": "12:30", "Event": "System maintenance completed", "Camera": "All", "Severity": "Info"},
        {"Time": "11:15", "Event": "High density detected", "Camera": "CAM_002", "Severity": "Warning"},
        {"Time": "10:00", "Event": "Daily system check", "Camera": "System", "Severity": "Info"}
    ]
    
    for activity in activity_data:
        severity_color = {
            "Warning": "#ffaa00",
            "Info": "#667eea",
            "Critical": "#ff4444"
        }.get(activity["Severity"], "#667eea")
        
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 8px; 
                    margin: 0.5rem 0; border-left: 4px solid {severity_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong>{activity["Event"]}</strong><br>
                    <small style="color: rgba(255,255,255,0.7);">Camera: {activity["Camera"]}</small>
                </div>
                <div style="text-align: right;">
                    <div style="color: {severity_color}; font-weight: bold;">{activity["Severity"]}</div>
                    <small style="color: rgba(255,255,255,0.7);">{activity["Time"]}</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)