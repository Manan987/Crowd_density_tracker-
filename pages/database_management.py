import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import json

def show_database_management():
    """Database management and monitoring interface"""
    st.title("🗄️ Database Management Center")
    
    # Get database service
    data_service = st.session_state.data_service
    
    # Database overview section
    show_database_overview(data_service)
    
    st.markdown("---")
    
    # Main database management tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Analytics", 
        "🔧 Configuration", 
        "📝 System Logs", 
        "⚠️ Alert History",
        "🧹 Maintenance"
    ])
    
    with tab1:
        show_data_analytics_tab(data_service)
    
    with tab2:
        show_configuration_tab(data_service)
    
    with tab3:
        show_system_logs_tab(data_service)
    
    with tab4:
        show_alert_history_tab(data_service)
    
    with tab5:
        show_maintenance_tab(data_service)

def show_database_overview(data_service):
    """Display database overview metrics"""
    st.markdown("## 📊 Database Overview")
    
    # Get database information
    db_info = data_service.get_database_info()
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="text-align: center;">
                <div style="font-size: 2.5rem; color: #4CAF50;">🗄️</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: white;">{db_info.get('database_type', 'PostgreSQL')}</div>
                <div style="color: rgba(255,255,255,0.8);">Database Type</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_records = db_info.get('total_records', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div style="text-align: center;">
                <div style="font-size: 2.5rem; color: #2196F3;">📈</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: white;">{total_records:,}</div>
                <div style="color: rgba(255,255,255,0.8);">Total Records</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        db_size = db_info.get('database_size', 'Unknown')
        st.markdown(f"""
        <div class="metric-card">
            <div style="text-align: center;">
                <div style="font-size: 2.5rem; color: #FF9800;">💾</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: white;">{db_size}</div>
                <div style="color: rgba(255,255,255,0.8);">Database Size</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        table_counts = db_info.get('table_counts', {})
        active_tables = len([t for t in table_counts.values() if t > 0])
        st.markdown(f"""
        <div class="metric-card">
            <div style="text-align: center;">
                <div style="font-size: 2.5rem; color: #9C27B0;">🏗️</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: white;">{active_tables}</div>
                <div style="color: rgba(255,255,255,0.8);">Active Tables</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Table breakdown
    st.markdown("### 📋 Table Statistics")
    if table_counts:
        df_tables = pd.DataFrame([
            {'Table': table, 'Record Count': count, 'Status': 'Active' if count > 0 else 'Empty'}
            for table, count in table_counts.items()
        ])
        
        # Create bar chart for table record counts
        fig = px.bar(
            df_tables, 
            x='Table', 
            y='Record Count',
            color='Status',
            title="Records per Table",
            color_discrete_map={'Active': '#4CAF50', 'Empty': '#F44336'}
        )
        fig.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

def show_data_analytics_tab(data_service):
    """Data analytics and insights tab"""
    st.markdown("### 📊 Data Analytics Dashboard")
    
    # Time range selection
    col1, col2 = st.columns(2)
    with col1:
        time_range = st.selectbox(
            "Analysis Period",
            ["Last 24 Hours", "Last 3 Days", "Last Week", "Last Month"],
            index=1
        )
    
    with col2:
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Density Trends", "Alert Patterns", "System Performance", "Camera Utilization"]
        )
    
    # Get time range in hours
    hours_map = {
        "Last 24 Hours": 24,
        "Last 3 Days": 72,
        "Last Week": 168,
        "Last Month": 720
    }
    hours = hours_map[time_range]
    
    if analysis_type == "Density Trends":
        show_density_trends_analysis(data_service, hours)
    elif analysis_type == "Alert Patterns":
        show_alert_patterns_analysis(data_service, hours)
    elif analysis_type == "System Performance":
        show_system_performance_analysis(data_service, hours)
    elif analysis_type == "Camera Utilization":
        show_camera_utilization_analysis(data_service, hours)

def show_density_trends_analysis(data_service, hours):
    """Show density trends analysis"""
    density_data = data_service.get_recent_density_data(hours=hours)
    
    if density_data.empty:
        st.info("No density data available for the selected period")
        return
    
    # Density over time chart
    fig = px.line(
        density_data,
        x='timestamp',
        y='density',
        color='camera_id',
        title=f"Density Trends - Last {hours} Hours"
    )
    fig.add_hline(y=2.0, line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
    fig.add_hline(y=4.0, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
    fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Statistical Summary**")
        stats = density_data['density'].describe()
        st.dataframe(stats.to_frame("Value"), use_container_width=True)
    
    with col2:
        st.markdown("**Peak Density Events**")
        peak_events = density_data.nlargest(5, 'density')[['timestamp', 'camera_id', 'density']]
        st.dataframe(peak_events, use_container_width=True)

def show_alert_patterns_analysis(data_service, hours):
    """Show alert patterns analysis"""
    alert_data = data_service.get_alert_events(hours=hours)
    
    if alert_data.empty:
        st.info("No alert events for the selected period")
        return
    
    # Alert frequency by camera
    col1, col2 = st.columns(2)
    
    with col1:
        camera_alerts = alert_data['camera_id'].value_counts()
        fig_bar = px.bar(
            x=camera_alerts.index,
            y=camera_alerts.values,
            title="Alerts by Camera",
            labels={'x': 'Camera ID', 'y': 'Alert Count'}
        )
        fig_bar.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        alert_types = alert_data['alert_level'].value_counts()
        fig_pie = px.pie(
            values=alert_types.values,
            names=alert_types.index,
            title="Alert Distribution by Type"
        )
        fig_pie.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Alert timeline
    alert_data['hour'] = alert_data['timestamp'].dt.hour
    hourly_alerts = alert_data.groupby('hour').size().reset_index(name='count')
    
    fig_timeline = px.bar(
        hourly_alerts,
        x='hour',
        y='count',
        title="Alert Frequency by Hour of Day"
    )
    fig_timeline.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_timeline, use_container_width=True)

def show_system_performance_analysis(data_service, hours):
    """Show system performance analysis"""
    st.markdown("**System Performance Metrics**")
    
    # Get system logs
    logs_data = data_service.get_system_logs(hours=hours)
    
    if not logs_data.empty:
        # Log level distribution
        log_levels = logs_data['level'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_logs = px.pie(
                values=log_levels.values,
                names=log_levels.index,
                title="System Log Distribution",
                color_discrete_map={
                    'INFO': '#4CAF50',
                    'WARNING': '#FF9800',
                    'ERROR': '#F44336',
                    'CRITICAL': '#9C27B0'
                }
            )
            fig_logs.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_logs, use_container_width=True)
        
        with col2:
            # Component activity
            component_activity = logs_data['component'].value_counts()
            fig_components = px.bar(
                x=component_activity.values,
                y=component_activity.index,
                orientation='h',
                title="Activity by Component"
            )
            fig_components.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_components, use_container_width=True)
        
        # Recent critical events
        critical_logs = logs_data[logs_data['level'].isin(['ERROR', 'CRITICAL'])].head(10)
        if not critical_logs.empty:
            st.markdown("**Recent Critical Events**")
            st.dataframe(
                critical_logs[['timestamp', 'level', 'component', 'message']],
                use_container_width=True
            )
    else:
        st.info("No system logs available for analysis")

def show_camera_utilization_analysis(data_service, hours):
    """Show camera utilization analysis"""
    camera_perf = data_service.get_camera_performance()
    
    if camera_perf:
        # Convert to DataFrame for easier plotting
        perf_data = []
        for camera_id, data in camera_perf.items():
            perf_data.append({
                'camera_id': camera_id,
                'name': data.get('name', camera_id),
                'total_readings': data.get('total_readings', 0),
                'average_density': data.get('average_density', 0),
                'uptime_percentage': data.get('uptime_percentage', 0),
                'status': data.get('status', 'unknown')
            })
        
        df_perf = pd.DataFrame(perf_data)
        
        # Camera performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            fig_uptime = px.bar(
                df_perf,
                x='camera_id',
                y='uptime_percentage',
                title="Camera Uptime (%)",
                color='uptime_percentage',
                color_continuous_scale='RdYlGn'
            )
            fig_uptime.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_uptime, use_container_width=True)
        
        with col2:
            fig_readings = px.bar(
                df_perf,
                x='camera_id',
                y='total_readings',
                title="Total Readings per Camera",
                color='status',
                color_discrete_map={'active': '#4CAF50', 'inactive': '#F44336'}
            )
            fig_readings.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_readings, use_container_width=True)
        
        # Camera details table
        st.markdown("**Camera Performance Details**")
        st.dataframe(df_perf, use_container_width=True)
    else:
        st.info("No camera performance data available")

def show_configuration_tab(data_service):
    """Configuration management tab"""
    st.markdown("### ⚙️ System Configuration")
    
    # Get current configuration
    alert_config = data_service.get_configuration(category="alert_thresholds")
    notification_config = data_service.get_configuration(category="notification_settings")
    model_config = data_service.get_configuration(category="model_config")
    
    # Alert thresholds configuration
    st.markdown("**Alert Threshold Settings**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        warning_threshold = st.number_input(
            "Warning Threshold (people/m²)",
            min_value=0.1,
            max_value=10.0,
            value=alert_config.get('alert_warning_threshold', 2.0),
            step=0.1
        )
    
    with col2:
        critical_threshold = st.number_input(
            "Critical Threshold (people/m²)",
            min_value=0.1,
            max_value=15.0,
            value=alert_config.get('alert_critical_threshold', 4.0),
            step=0.1
        )
    
    with col3:
        alert_duration = st.number_input(
            "Alert Duration (seconds)",
            min_value=1,
            max_value=300,
            value=alert_config.get('alert_duration_seconds', 30),
            step=1
        )
    
    if st.button("Update Alert Settings", type="primary"):
        success = True
        success &= data_service.update_configuration('alert_warning_threshold', warning_threshold)
        success &= data_service.update_configuration('alert_critical_threshold', critical_threshold)
        success &= data_service.update_configuration('alert_duration_seconds', alert_duration)
        
        if success:
            st.success("Alert settings updated successfully!")
        else:
            st.error("Failed to update some settings")
    
    st.markdown("---")
    
    # Model configuration
    st.markdown("**AI Model Settings**")
    col1, col2 = st.columns(2)
    
    with col1:
        confidence_threshold = st.slider(
            "Model Confidence Threshold",
            min_value=0.5,
            max_value=1.0,
            value=model_config.get('model_confidence_threshold', 0.85),
            step=0.05
        )
    
    with col2:
        enable_notifications = st.checkbox(
            "Enable Email Notifications",
            value=notification_config.get('email_notifications_enabled', True)
        )
    
    if st.button("Update Model Settings"):
        success = True
        success &= data_service.update_configuration('model_confidence_threshold', confidence_threshold)
        success &= data_service.update_configuration('email_notifications_enabled', enable_notifications)
        
        if success:
            st.success("Model settings updated successfully!")
        else:
            st.error("Failed to update model settings")

def show_system_logs_tab(data_service):
    """System logs management tab"""
    st.markdown("### 📝 System Logs")
    
    # Log filtering options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        log_hours = st.selectbox(
            "Time Period",
            [1, 6, 24, 72, 168],
            format_func=lambda x: f"Last {x} hours",
            index=2
        )
    
    with col2:
        log_level = st.selectbox(
            "Log Level",
            [None, "INFO", "WARNING", "ERROR", "CRITICAL"],
            format_func=lambda x: "All Levels" if x is None else x
        )
    
    with col3:
        log_component = st.selectbox(
            "Component",
            [None, "AI_MODEL", "VIDEO_PROCESSOR", "ALERT_SYSTEM", "DATA_SERVICE"],
            format_func=lambda x: "All Components" if x is None else x
        )
    
    # Get logs
    logs_data = data_service.get_system_logs(hours=log_hours, level=log_level, component=log_component)
    
    if not logs_data.empty:
        # Log level counts
        level_counts = logs_data['level'].value_counts()
        
        # Display log level distribution
        col1, col2, col3, col4 = st.columns(4)
        for i, (level, count) in enumerate(level_counts.items()):
            color_map = {
                'INFO': '#4CAF50',
                'WARNING': '#FF9800', 
                'ERROR': '#F44336',
                'CRITICAL': '#9C27B0'
            }
            
            with [col1, col2, col3, col4][i % 4]:
                st.markdown(f"""
                <div style="background: {color_map.get(level, '#666')}20; padding: 1rem; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: bold; color: {color_map.get(level, '#666')};">{count}</div>
                    <div style="color: rgba(255,255,255,0.8);">{level}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Display logs table
        st.markdown("**Recent System Logs**")
        display_logs = logs_data.copy()
        display_logs['timestamp'] = display_logs['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        st.dataframe(
            display_logs[['timestamp', 'level', 'component', 'message', 'camera_id']],
            use_container_width=True,
            height=400
        )
        
        # Export logs
        if st.button("📤 Export Logs"):
            csv_data = display_logs.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"system_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No logs found matching the selected criteria")

def show_alert_history_tab(data_service):
    """Alert history management tab"""
    st.markdown("### ⚠️ Alert History")
    
    # Get alert events
    alert_data = data_service.get_alert_events(hours=168)  # Last week
    
    if not alert_data.empty:
        # Alert summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_alerts = len(alert_data)
        critical_alerts = len(alert_data[alert_data['alert_level'] == 'Critical'])
        warning_alerts = len(alert_data[alert_data['alert_level'] == 'Warning'])
        acknowledged_alerts = len(alert_data[alert_data['acknowledged'] == True])
        
        with col1:
            st.metric("Total Alerts", total_alerts)
        with col2:
            st.metric("Critical Alerts", critical_alerts)
        with col3:
            st.metric("Warning Alerts", warning_alerts)
        with col4:
            ack_rate = (acknowledged_alerts / total_alerts * 100) if total_alerts > 0 else 0
            st.metric("Acknowledgment Rate", f"{ack_rate:.1f}%")
        
        # Alert timeline
        fig_timeline = px.scatter(
            alert_data,
            x='timestamp',
            y='camera_id',
            color='alert_level',
            size='density',
            title="Alert Timeline",
            color_discrete_map={'Warning': 'orange', 'Critical': 'red'}
        )
        fig_timeline.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Unacknowledged alerts
        unack_alerts = alert_data[alert_data['acknowledged'] == False]
        if not unack_alerts.empty:
            st.markdown("**Unacknowledged Alerts**")
            
            for _, alert in unack_alerts.head(5).iterrows():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    level_color = "#ff4757" if alert['alert_level'] == 'Critical' else "#ffa726"
                    st.markdown(f"""
                    <div style="background: {level_color}20; border-left: 4px solid {level_color}; 
                                padding: 1rem; margin: 0.5rem 0; border-radius: 4px;">
                        <strong>{alert['alert_level']} Alert - {alert['camera_id']}</strong><br>
                        Density: {alert['density']:.2f} people/m² | {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if st.button(f"Acknowledge", key=f"ack_{alert['id']}"):
                        success = data_service.acknowledge_alert(alert['id'], "operator")
                        if success:
                            st.success("Alert acknowledged!")
                            st.experimental_rerun()
        
        # Alert history table
        st.markdown("**Complete Alert History**")
        display_alerts = alert_data.copy()
        display_alerts['timestamp'] = display_alerts['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        st.dataframe(
            display_alerts[['timestamp', 'camera_id', 'alert_level', 'density', 'acknowledged']],
            use_container_width=True,
            height=300
        )
    else:
        st.info("No alert history available")

def show_maintenance_tab(data_service):
    """Database maintenance tab"""
    st.markdown("### 🧹 Database Maintenance")
    
    # Database statistics
    db_info = data_service.get_database_info()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Database Health**")
        st.json(db_info)
    
    with col2:
        st.markdown("**Maintenance Operations**")
        
        # Data cleanup
        cleanup_days = st.number_input(
            "Delete data older than (days)",
            min_value=1,
            max_value=365,
            value=30,
            step=1
        )
        
        if st.button("🗑️ Clean Old Data", type="secondary"):
            if st.checkbox("I confirm this action"):
                try:
                    data_service.clear_old_data(days=cleanup_days)
                    st.success(f"Cleaned data older than {cleanup_days} days")
                except Exception as e:
                    st.error(f"Cleanup failed: {e}")
        
        # Data export
        st.markdown("**Data Export**")
        export_format = st.selectbox("Export Format", ["CSV", "JSON"])
        export_hours = st.selectbox("Export Period", [24, 72, 168, 720])
        
        if st.button("📤 Export Data"):
            try:
                exported_data = data_service.export_data(
                    hours=export_hours, 
                    format=export_format.lower()
                )
                
                filename = f"crowdguard_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format.lower()}"
                mime_type = "text/csv" if export_format == "CSV" else "application/json"
                
                st.download_button(
                    label=f"Download {export_format}",
                    data=exported_data,
                    file_name=filename,
                    mime=mime_type
                )
            except Exception as e:
                st.error(f"Export failed: {e}")