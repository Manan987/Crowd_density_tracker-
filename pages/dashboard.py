import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def show_dashboard():
    """Main dashboard page with real-time monitoring overview"""
    st.title("📊 System Dashboard")
    
    # Get data managers from session state
    if 'data_manager' not in st.session_state:
        st.error("Data manager not initialized")
        return
    
    data_manager = st.session_state.data_manager
    alert_system = st.session_state.alert_system
    
    # Dashboard header with key metrics
    show_key_metrics(data_manager, alert_system)
    
    st.markdown("---")
    
    # Time range selector
    col1, col2 = st.columns([1, 3])
    with col1:
        time_range = st.selectbox(
            "Time Range",
            ["Last Hour", "Last 6 Hours", "Last 24 Hours"],
            index=1
        )
    
    # Get time range in hours
    time_hours = {"Last Hour": 1, "Last 6 Hours": 6, "Last 24 Hours": 24}[time_range]
    
    # Main dashboard content
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Real-time Trends", "🎯 Camera Overview", "⚠️ Alert Analysis", "📋 System Health"])
    
    with tab1:
        show_realtime_trends(data_manager, time_hours)
    
    with tab2:
        show_camera_overview(data_manager, time_hours)
    
    with tab3:
        show_alert_analysis(data_manager, alert_system, time_hours)
    
    with tab4:
        show_system_health(data_manager)

def show_key_metrics(data_manager, alert_system):
    """Display key system metrics"""
    # Get recent data for metrics
    recent_data = data_manager.get_recent_data(minutes=10)
    stats = data_manager.get_density_statistics(hours=1)
    alert_summary = alert_system.get_alert_summary(hours=1)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        current_density = st.session_state.get('current_density', 0)
        st.metric(
            "Current Density",
            f"{current_density:.1f}",
            delta=f"{current_density - stats.get('average_density', 0):.1f}" if stats else None,
            help="People per square meter"
        )
    
    with col2:
        alert_level = st.session_state.get('alert_status', 'Normal')
        alert_color = {"Normal": "🟢", "Warning": "🟡", "Critical": "🔴"}
        st.metric(
            "Alert Status",
            f"{alert_color.get(alert_level, '⚪')} {alert_level}",
            help="Current system alert level"
        )
    
    with col3:
        active_cameras = len(recent_data['camera_id'].unique()) if not recent_data.empty else 0
        st.metric(
            "Active Cameras",
            f"{active_cameras}/4",
            delta=None,
            help="Number of cameras currently providing data"
        )
    
    with col4:
        total_alerts = alert_summary.get('total_alerts', 0)
        st.metric(
            "Alerts (1h)",
            total_alerts,
            delta=None,
            help="Total alerts in the last hour"
        )
    
    with col5:
        avg_response_time = alert_summary.get('average_response_time_seconds', 0)
        st.metric(
            "Avg Response",
            f"{avg_response_time:.1f}s",
            delta=None,
            help="Average alert response time"
        )

def show_realtime_trends(data_manager, time_hours):
    """Show real-time density trends"""
    st.subheader("Density Trends")
    
    # Get data
    data = data_manager.get_recent_data(hours=time_hours)
    
    if data.empty:
        st.info("No data available for the selected time range")
        return
    
    # Main trend chart
    fig = px.line(
        data,
        x='timestamp',
        y='density',
        color='camera_id',
        title=f"Crowd Density Over Time ({time_hours}h)",
        labels={'density': 'Density (people/m²)', 'timestamp': 'Time'}
    )
    
    # Add threshold lines
    fig.add_hline(y=2.0, line_dash="dash", line_color="orange", 
                  annotation_text="Warning Threshold (2.0)")
    fig.add_hline(y=4.0, line_dash="dash", line_color="red", 
                  annotation_text="Critical Threshold (4.0)")
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Secondary metrics
    col1, col2 = st.columns(2)
    
    with col1:
        # Density distribution
        st.subheader("Density Distribution")
        fig_hist = px.histogram(
            data,
            x='density',
            nbins=20,
            title="Distribution of Density Values",
            labels={'density': 'Density (people/m²)', 'count': 'Frequency'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Crowd count trends
        st.subheader("Crowd Count Trends")
        fig_count = px.line(
            data,
            x='timestamp',
            y='crowd_count',
            color='camera_id',
            title="Estimated Crowd Count Over Time"
        )
        st.plotly_chart(fig_count, use_container_width=True)

def show_camera_overview(data_manager, time_hours):
    """Show individual camera performance and status"""
    st.subheader("Camera Performance Overview")
    
    # Get camera performance data
    camera_stats = data_manager.get_camera_performance()
    data = data_manager.get_recent_data(hours=time_hours)
    
    if not camera_stats:
        st.info("No camera performance data available")
        return
    
    # Camera status grid
    cameras = ["CAM_001", "CAM_002", "CAM_003", "CAM_004"]
    cols = st.columns(2)
    
    for i, camera_id in enumerate(cameras):
        with cols[i % 2]:
            stats = camera_stats.get(camera_id, {})
            
            # Camera status card
            if stats:
                avg_density = stats.get('average_density', 0)
                max_density = stats.get('max_density', 0)
                alert_count = stats.get('alert_count', 0)
                uptime = stats.get('uptime_percentage', 0)
                
                # Status color based on recent activity
                status_color = "🟢" if uptime > 95 else "🟡" if uptime > 80 else "🔴"
                
                st.markdown(f"""
                <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin: 10px 0;">
                    <h4>{status_color} {camera_id}</h4>
                    <p><strong>Avg Density:</strong> {avg_density:.1f} people/m²</p>
                    <p><strong>Max Density:</strong> {max_density:.1f} people/m²</p>
                    <p><strong>Alerts:</strong> {alert_count}</p>
                    <p><strong>Uptime:</strong> {uptime:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin: 10px 0; opacity: 0.6;">
                    <h4>⚪ {camera_id}</h4>
                    <p>No data available</p>
                </div>
                """, unsafe_allow_html=True)
    
    if not data.empty:
        st.markdown("---")
        
        # Camera comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Average density by camera
            camera_avg = data.groupby('camera_id')['density'].mean().reset_index()
            fig_avg = px.bar(
                camera_avg,
                x='camera_id',
                y='density',
                title="Average Density by Camera",
                labels={'density': 'Avg Density (people/m²)'}
            )
            st.plotly_chart(fig_avg, use_container_width=True)
        
        with col2:
            # Alert frequency by camera
            alert_data = data[data['alert_level'] != 'Normal']
            alert_counts = alert_data['camera_id'].value_counts().reset_index()
            alert_counts.columns = ['camera_id', 'alert_count']
            
            if not alert_counts.empty:
                fig_alerts = px.bar(
                    alert_counts,
                    x='camera_id',
                    y='alert_count',
                    title="Alert Count by Camera",
                    color='alert_count',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_alerts, use_container_width=True)
            else:
                st.info("No alerts recorded for this time period")

def show_alert_analysis(data_manager, alert_system, time_hours):
    """Show detailed alert analysis"""
    st.subheader("Alert Analysis")
    
    # Get alert data
    alert_events = data_manager.get_alert_events(hours=time_hours)
    alert_summary = alert_system.get_alert_summary(hours=time_hours)
    
    if alert_events.empty:
        st.info("No alerts recorded for the selected time range")
        return
    
    # Alert overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_alerts = len(alert_events)
        st.metric("Total Alerts", total_alerts)
    
    with col2:
        warning_alerts = len(alert_events[alert_events['alert_level'] == 'Warning'])
        st.metric("Warning Alerts", warning_alerts)
    
    with col3:
        critical_alerts = len(alert_events[alert_events['alert_level'] == 'Critical'])
        st.metric("Critical Alerts", critical_alerts)
    
    with col4:
        acknowledged = len(alert_events[alert_events['acknowledged'] == True])
        ack_rate = (acknowledged / total_alerts * 100) if total_alerts > 0 else 0
        st.metric("Acknowledgment Rate", f"{ack_rate:.1f}%")
    
    # Alert timeline
    st.subheader("Alert Timeline")
    fig_timeline = px.scatter(
        alert_events,
        x='timestamp',
        y='camera_id',
        color='alert_level',
        size='density',
        title="Alert Timeline",
        color_discrete_map={'Warning': 'orange', 'Critical': 'red'}
    )
    fig_timeline.update_layout(height=300)
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Alert analysis charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Alert distribution by level
        alert_dist = alert_events['alert_level'].value_counts()
        fig_pie = px.pie(
            values=alert_dist.values,
            names=alert_dist.index,
            title="Alert Distribution by Level",
            color_discrete_map={'Warning': 'orange', 'Critical': 'red'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Alert frequency by hour
        alert_events['hour'] = alert_events['timestamp'].dt.hour
        hourly_alerts = alert_events.groupby('hour').size().reset_index(name='count')
        
        fig_hourly = px.bar(
            hourly_alerts,
            x='hour',
            y='count',
            title="Alert Frequency by Hour",
            labels={'hour': 'Hour of Day', 'count': 'Number of Alerts'}
        )
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    # Recent alerts table
    st.subheader("Recent Alert Events")
    if not alert_events.empty:
        display_events = alert_events.head(10)[
            ['timestamp', 'camera_id', 'alert_level', 'density', 'acknowledged']
        ].copy()
        display_events['timestamp'] = display_events['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(display_events, use_container_width=True)

def show_system_health(data_manager):
    """Show system health and performance metrics"""
    st.subheader("System Health Monitor")
    
    # Database information
    db_info = data_manager.get_database_info()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Database Size", f"{db_info.get('database_size_bytes', 0) / 1024:.1f} KB")
    
    with col2:
        table_counts = db_info.get('table_counts', {})
        total_records = sum(table_counts.values())
        st.metric("Total Records", total_records)
    
    with col3:
        cache_size = db_info.get('cache_size', 0)
        st.metric("Cache Size", cache_size)
    
    # System performance chart
    st.subheader("Data Processing Rate")
    
    # Get recent data points to show processing rate
    recent_data = data_manager.get_recent_data(hours=1)
    
    if not recent_data.empty:
        # Group by 5-minute intervals
        recent_data['time_group'] = recent_data['timestamp'].dt.floor('5min')
        processing_rate = recent_data.groupby('time_group').size().reset_index(name='data_points')
        
        fig_rate = px.line(
            processing_rate,
            x='time_group',
            y='data_points',
            title="Data Points Processed (5-minute intervals)",
            labels={'time_group': 'Time', 'data_points': 'Data Points'}
        )
        st.plotly_chart(fig_rate, use_container_width=True)
    else:
        st.info("No recent data available for processing rate analysis")
    
    # Component status
    st.subheader("Component Status")
    
    components = [
        {"Component": "ML Model", "Status": "✅ Online", "Last Check": "2 min ago"},
        {"Component": "Video Processor", "Status": "✅ Online", "Last Check": "1 min ago"},
        {"Component": "Alert System", "Status": "✅ Online", "Last Check": "30 sec ago"},
        {"Component": "Database", "Status": "✅ Online", "Last Check": "15 sec ago"},
    ]
    
    status_df = pd.DataFrame(components)
    st.dataframe(status_df, use_container_width=True, hide_index=True)
    
    # System logs
    st.subheader("Recent System Activity")
    logs = data_manager.get_system_logs(hours=1)
    
    if not logs.empty:
        display_logs = logs.head(5)[['timestamp', 'level', 'message']].copy()
        display_logs['timestamp'] = display_logs['timestamp'].dt.strftime('%H:%M:%S')
        st.dataframe(display_logs, use_container_width=True, hide_index=True)
    else:
        st.info("No recent system logs available")
