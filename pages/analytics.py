import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import math

def show_analytics():
    """Advanced analytics and reporting page"""
    st.title("📊 Advanced Analytics & Reports")
    
    # Get data managers from session state
    if 'data_manager' not in st.session_state:
        st.error("Data manager not initialized")
        return
    
    data_manager = st.session_state.data_manager
    alert_system = st.session_state.alert_system
    
    # Analytics control panel
    show_analytics_controls()
    
    st.markdown("---")
    
    # Main analytics tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Trend Analysis", 
        "🔍 Pattern Detection", 
        "📊 Statistical Reports", 
        "🎯 Predictive Analytics",
        "📋 Custom Reports"
    ])
    
    with tab1:
        show_trend_analysis(data_manager)
    
    with tab2:
        show_pattern_detection(data_manager)
    
    with tab3:
        show_statistical_reports(data_manager, alert_system)
    
    with tab4:
        show_predictive_analytics(data_manager)
    
    with tab5:
        show_custom_reports(data_manager)

def show_analytics_controls():
    """Analytics control panel for filtering and configuration"""
    st.subheader("🎛️ Analytics Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Time range selection
        time_range = st.selectbox(
            "📅 Time Range",
            ["Last 6 Hours", "Last 24 Hours", "Last 3 Days", 
             "Last Week", "Last Month", "Custom Range"],
            index=2,
            key="analytics_time_range"
        )
        
        if time_range == "Custom Range":
            start_date = st.date_input("Start Date", key="analytics_start_date")
            end_date = st.date_input("End Date", key="analytics_end_date")
    
    with col2:
        # Camera selection
        cameras = ["All Cameras", "CAM_001", "CAM_002", "CAM_003", "CAM_004"]
        selected_cameras = st.multiselect(
            "📹 Camera Filter",
            cameras[1:],  # Exclude "All Cameras" from options
            default=cameras[1:],  # Select all by default
            key="analytics_cameras"
        )
        
        if not selected_cameras:
            selected_cameras = cameras[1:]  # Default to all if none selected
    
    with col3:
        # Analysis granularity
        granularity = st.selectbox(
            "⏰ Data Granularity",
            ["1 Minute", "5 Minutes", "15 Minutes", "1 Hour"],
            index=1,
            key="analytics_granularity"
        )
        
        # Alert level filter
        alert_filter = st.selectbox(
            "🚨 Alert Level Filter",
            ["All Levels", "Normal Only", "Warning+", "Critical Only"],
            key="analytics_alert_filter"
        )
    
    with col4:
        # Analysis options
        include_weekends = st.checkbox(
            "Include Weekends",
            value=True,
            key="analytics_weekends"
        )
        
        smooth_data = st.checkbox(
            "Smooth Data",
            value=False,
            help="Apply smoothing to reduce noise in visualizations",
            key="analytics_smooth"
        )
    
    # Store selections in session state for use across functions
    st.session_state.analytics_config = {
        'time_range': time_range,
        'cameras': selected_cameras,
        'granularity': granularity,
        'alert_filter': alert_filter,
        'include_weekends': include_weekends,
        'smooth_data': smooth_data
    }

def get_filtered_data(data_manager):
    """Get data based on analytics control settings"""
    config = st.session_state.get('analytics_config', {})
    
    # Calculate time range
    time_range = config.get('time_range', 'Last 24 Hours')
    
    if time_range == "Custom Range":
        # Custom range logic would go here
        hours = 24
    else:
        hours_map = {
            "Last 6 Hours": 6,
            "Last 24 Hours": 24,
            "Last 3 Days": 72,
            "Last Week": 168,
            "Last Month": 720
        }
        hours = hours_map.get(time_range, 24)
    
    # Get data
    data = data_manager.get_recent_data(
        hours=hours,
        cameras=config.get('cameras', ['CAM_001', 'CAM_002', 'CAM_003', 'CAM_004'])
    )
    
    if data.empty:
        return data
    
    # Apply alert level filter
    alert_filter = config.get('alert_filter', 'All Levels')
    if alert_filter == "Normal Only":
        data = data[data['alert_level'] == 'Normal']
    elif alert_filter == "Warning+":
        data = data[data['alert_level'].isin(['Warning', 'Critical'])]
    elif alert_filter == "Critical Only":
        data = data[data['alert_level'] == 'Critical']
    
    # Apply weekend filter
    if not config.get('include_weekends', True):
        data['weekday'] = data['timestamp'].dt.weekday
        data = data[data['weekday'] < 5]  # Monday=0, Sunday=6
        data = data.drop('weekday', axis=1)
    
    # Apply data smoothing if requested
    if config.get('smooth_data', False) and len(data) > 5:
        data = apply_data_smoothing(data)
    
    return data

def apply_data_smoothing(data):
    """Apply smoothing to density data"""
    try:
        # Sort by timestamp and camera
        data = data.sort_values(['camera_id', 'timestamp'])
        
        # Apply rolling average for each camera
        smoothed_data = []
        for camera in data['camera_id'].unique():
            camera_data = data[data['camera_id'] == camera].copy()
            camera_data['density'] = camera_data['density'].rolling(window=5, center=True, min_periods=1).mean()
            smoothed_data.append(camera_data)
        
        return pd.concat(smoothed_data, ignore_index=True)
    except Exception as e:
        st.warning(f"Could not apply smoothing: {e}")
        return data

def show_trend_analysis(data_manager):
    """Advanced trend analysis with multiple visualization options"""
    st.header("📈 Trend Analysis")
    
    data = get_filtered_data(data_manager)
    
    if data.empty:
        st.warning("No data available for the selected filters. Try adjusting your time range or camera selection.")
        return
    
    # Trend analysis options
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Analysis Options")
        
        trend_type = st.selectbox(
            "Trend Analysis Type",
            ["Linear Trend", "Moving Average", "Seasonal Decomposition", "Growth Rate"],
            key="trend_type"
        )
        
        if trend_type == "Moving Average":
            ma_window = st.slider("Moving Average Window", 3, 50, 10)
        
        show_confidence_bands = st.checkbox("Show Confidence Bands", value=False)
        
        if show_confidence_bands:
            confidence_level = st.slider("Confidence Level", 0.8, 0.99, 0.95, 0.01)
    
    with col1:
        st.subheader("Density Trends Over Time")
        
        # Create trend visualization based on selected type
        if trend_type == "Linear Trend":
            fig = create_linear_trend_chart(data, show_confidence_bands, 
                                          confidence_level if show_confidence_bands else 0.95)
        elif trend_type == "Moving Average":
            fig = create_moving_average_chart(data, ma_window)
        elif trend_type == "Seasonal Decomposition":
            fig = create_seasonal_decomposition_chart(data)
        else:  # Growth Rate
            fig = create_growth_rate_chart(data)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional trend insights
    st.markdown("---")
    st.subheader("Trend Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Peak hours analysis
        st.write("**Peak Hours Analysis**")
        peak_hours = analyze_peak_hours(data)
        for camera, hours in peak_hours.items():
            st.write(f"📹 {camera}: {hours}")
    
    with col2:
        # Trend direction
        st.write("**Trend Direction**")
        trend_direction = analyze_trend_direction(data)
        for camera, direction in trend_direction.items():
            direction_emoji = "📈" if direction > 0 else "📉" if direction < 0 else "➡️"
            st.write(f"📹 {camera}: {direction_emoji} {direction:.3f}")
    
    with col3:
        # Volatility analysis
        st.write("**Density Volatility**")
        volatility = analyze_volatility(data)
        for camera, vol in volatility.items():
            vol_level = "High" if vol > 1.0 else "Medium" if vol > 0.5 else "Low"
            st.write(f"📹 {camera}: {vol_level} ({vol:.2f})")

def create_linear_trend_chart(data, show_confidence_bands, confidence_level):
    """Create linear trend chart with optional confidence bands"""
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, camera in enumerate(data['camera_id'].unique()):
        camera_data = data[data['camera_id'] == camera].sort_values('timestamp')
        
        if len(camera_data) < 2:
            continue
        
        # Convert timestamps to numeric for regression
        x_numeric = pd.to_numeric(camera_data['timestamp'])
        y = camera_data['density']
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregr(x_numeric, y)
        trend_line = slope * x_numeric + intercept
        
        # Original data
        fig.add_trace(go.Scatter(
            x=camera_data['timestamp'],
            y=y,
            mode='lines+markers',
            name=f'{camera} Data',
            line=dict(color=colors[i % len(colors)]),
            opacity=0.7
        ))
        
        # Trend line
        fig.add_trace(go.Scatter(
            x=camera_data['timestamp'],
            y=trend_line,
            mode='lines',
            name=f'{camera} Trend',
            line=dict(color=colors[i % len(colors)], width=3, dash='dash')
        ))
        
        # Confidence bands
        if show_confidence_bands and len(camera_data) > 3:
            residuals = y - trend_line
            mse = np.mean(residuals**2)
            t_val = stats.t.ppf((1 + confidence_level) / 2, len(camera_data) - 2)
            margin_error = t_val * np.sqrt(mse * (1 + 1/len(camera_data)))
            
            fig.add_trace(go.Scatter(
                x=camera_data['timestamp'],
                y=trend_line + margin_error,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=camera_data['timestamp'],
                y=trend_line - margin_error,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor=f'rgba{colors[i % len(colors)][3:-1]}, 0.1)',
                name=f'{camera} Confidence Band',
                hoverinfo='skip'
            ))
    
    fig.update_layout(
        title="Linear Trend Analysis",
        xaxis_title="Time",
        yaxis_title="Density (people/m²)",
        height=500
    )
    
    return fig

def create_moving_average_chart(data, window):
    """Create moving average trend chart"""
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, camera in enumerate(data['camera_id'].unique()):
        camera_data = data[data['camera_id'] == camera].sort_values('timestamp')
        
        if len(camera_data) < window:
            continue
        
        # Calculate moving average
        camera_data['ma'] = camera_data['density'].rolling(window=window, center=True).mean()
        
        # Original data
        fig.add_trace(go.Scatter(
            x=camera_data['timestamp'],
            y=camera_data['density'],
            mode='lines',
            name=f'{camera} Data',
            line=dict(color=colors[i % len(colors)], width=1),
            opacity=0.5
        ))
        
        # Moving average
        fig.add_trace(go.Scatter(
            x=camera_data['timestamp'],
            y=camera_data['ma'],
            mode='lines',
            name=f'{camera} MA({window})',
            line=dict(color=colors[i % len(colors)], width=3)
        ))
    
    fig.update_layout(
        title=f"Moving Average Trend (Window: {window})",
        xaxis_title="Time",
        yaxis_title="Density (people/m²)",
        height=500
    )
    
    return fig

def create_seasonal_decomposition_chart(data):
    """Create seasonal decomposition visualization"""
    # This is a simplified version - full seasonal decomposition would require more sophisticated analysis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Hourly Pattern', 'Daily Pattern', 'Weekly Pattern', 'Overall Trend'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    if len(data) < 24:  # Need sufficient data for seasonal analysis
        fig.add_annotation(
            text="Insufficient data for seasonal decomposition",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Add hour, day, and weekday columns
    data = data.copy()
    data['hour'] = data['timestamp'].dt.hour
    data['day'] = data['timestamp'].dt.day
    data['weekday'] = data['timestamp'].dt.day_name()
    
    # Hourly pattern
    hourly_pattern = data.groupby('hour')['density'].mean()
    fig.add_trace(
        go.Scatter(x=hourly_pattern.index, y=hourly_pattern.values, mode='lines+markers'),
        row=1, col=1
    )
    
    # Daily pattern (if we have enough days)
    if len(data['day'].unique()) > 1:
        daily_pattern = data.groupby('day')['density'].mean()
        fig.add_trace(
            go.Scatter(x=daily_pattern.index, y=daily_pattern.values, mode='lines+markers'),
            row=1, col=2
        )
    
    # Weekly pattern
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_pattern = data.groupby('weekday')['density'].mean().reindex(weekday_order)
    fig.add_trace(
        go.Bar(x=weekday_pattern.index, y=weekday_pattern.values),
        row=2, col=1
    )
    
    # Overall trend
    daily_avg = data.groupby(data['timestamp'].dt.date)['density'].mean()
    fig.add_trace(
        go.Scatter(x=daily_avg.index, y=daily_avg.values, mode='lines'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    return fig

def create_growth_rate_chart(data):
    """Create growth rate analysis chart"""
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, camera in enumerate(data['camera_id'].unique()):
        camera_data = data[data['camera_id'] == camera].sort_values('timestamp')
        
        if len(camera_data) < 2:
            continue
        
        # Calculate percentage change
        camera_data['pct_change'] = camera_data['density'].pct_change() * 100
        
        fig.add_trace(go.Scatter(
            x=camera_data['timestamp'],
            y=camera_data['pct_change'],
            mode='lines',
            name=f'{camera} Growth Rate',
            line=dict(color=colors[i % len(colors)])
        ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    
    fig.update_layout(
        title="Density Growth Rate (%)",
        xaxis_title="Time",
        yaxis_title="Growth Rate (%)",
        height=500
    )
    
    return fig

def analyze_peak_hours(data):
    """Analyze peak hours for each camera"""
    peak_hours = {}
    
    for camera in data['camera_id'].unique():
        camera_data = data[data['camera_id'] == camera]
        if camera_data.empty:
            continue
        
        camera_data = camera_data.copy()
        camera_data['hour'] = camera_data['timestamp'].dt.hour
        hourly_avg = camera_data.groupby('hour')['density'].mean()
        
        if not hourly_avg.empty:
            peak_hour = hourly_avg.idxmax()
            peak_density = hourly_avg.max()
            peak_hours[camera] = f"{peak_hour:02d}:00 ({peak_density:.1f})"
    
    return peak_hours

def analyze_trend_direction(data):
    """Analyze overall trend direction for each camera"""
    trend_direction = {}
    
    for camera in data['camera_id'].unique():
        camera_data = data[data['camera_id'] == camera].sort_values('timestamp')
        if len(camera_data) < 2:
            continue
        
        # Calculate linear trend slope
        x_numeric = pd.to_numeric(camera_data['timestamp'])
        y = camera_data['density']
        
        slope, _, _, _, _ = stats.linregr(x_numeric, y)
        trend_direction[camera] = slope
    
    return trend_direction

def analyze_volatility(data):
    """Analyze density volatility for each camera"""
    volatility = {}
    
    for camera in data['camera_id'].unique():
        camera_data = data[data['camera_id'] == camera]
        if camera_data.empty:
            continue
        
        # Calculate coefficient of variation as volatility measure
        std_dev = camera_data['density'].std()
        mean_density = camera_data['density'].mean()
        
        if mean_density > 0:
            cv = std_dev / mean_density
            volatility[camera] = cv
    
    return volatility

def show_pattern_detection(data_manager):
    """Advanced pattern detection and anomaly analysis"""
    st.header("🔍 Pattern Detection")
    
    data = get_filtered_data(data_manager)
    
    if data.empty:
        st.warning("No data available for pattern detection.")
        return
    
    # Pattern detection options
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Detection Settings")
        
        pattern_type = st.selectbox(
            "Pattern Type",
            ["Anomaly Detection", "Recurring Patterns", "Correlation Analysis", "Clustering"],
            key="pattern_type"
        )
        
        sensitivity = st.slider(
            "Detection Sensitivity",
            1, 10, 5,
            help="Higher values detect more patterns/anomalies"
        )
        
        min_pattern_length = st.number_input(
            "Minimum Pattern Length",
            1, 50, 5,
            help="Minimum number of data points for a pattern"
        )
    
    with col1:
        if pattern_type == "Anomaly Detection":
            show_anomaly_detection(data, sensitivity)
        elif pattern_type == "Recurring Patterns":
            show_recurring_patterns(data, min_pattern_length)
        elif pattern_type == "Correlation Analysis":
            show_correlation_analysis(data)
        else:  # Clustering
            show_clustering_analysis(data)

def show_anomaly_detection(data, sensitivity):
    """Show anomaly detection results"""
    st.subheader("Anomaly Detection Results")
    
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    anomaly_threshold = 2.0 + (10 - sensitivity) * 0.3  # Adjust threshold based on sensitivity
    
    anomalies_found = []
    
    for i, camera in enumerate(data['camera_id'].unique()):
        camera_data = data[data['camera_id'] == camera].sort_values('timestamp')
        
        if len(camera_data) < 10:
            continue
        
        # Calculate z-scores for anomaly detection
        camera_data = camera_data.copy()
        camera_data['z_score'] = np.abs((camera_data['density'] - camera_data['density'].mean()) / camera_data['density'].std())
        
        # Identify anomalies
        anomalies = camera_data[camera_data['z_score'] > anomaly_threshold]
        
        # Plot normal data
        normal_data = camera_data[camera_data['z_score'] <= anomaly_threshold]
        fig.add_trace(go.Scatter(
            x=normal_data['timestamp'],
            y=normal_data['density'],
            mode='lines+markers',
            name=f'{camera} Normal',
            line=dict(color=colors[i % len(colors)]),
            marker=dict(size=4)
        ))
        
        # Plot anomalies
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies['timestamp'],
                y=anomalies['density'],
                mode='markers',
                name=f'{camera} Anomalies',
                marker=dict(
                    color='red',
                    size=8,
                    symbol='x'
                )
            ))
            
            anomalies_found.extend([
                {
                    'camera': camera,
                    'timestamp': row['timestamp'],
                    'density': row['density'],
                    'z_score': row['z_score']
                }
                for _, row in anomalies.iterrows()
            ])
    
    fig.update_layout(
        title="Anomaly Detection in Crowd Density",
        xaxis_title="Time",
        yaxis_title="Density (people/m²)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Anomaly summary
    if anomalies_found:
        st.subheader(f"Found {len(anomalies_found)} Anomalies")
        
        anomalies_df = pd.DataFrame(anomalies_found)
        anomalies_df['timestamp'] = anomalies_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        anomalies_df = anomalies_df.round({'density': 2, 'z_score': 2})
        
        st.dataframe(anomalies_df, use_container_width=True)
    else:
        st.info("No anomalies detected with current sensitivity settings.")

def show_recurring_patterns(data, min_length):
    """Show recurring pattern analysis"""
    st.subheader("Recurring Pattern Analysis")
    
    # Analyze hourly patterns
    data_copy = data.copy()
    data_copy['hour'] = data_copy['timestamp'].dt.hour
    data_copy['weekday'] = data_copy['timestamp'].dt.day_name()
    
    # Create heatmap of density by hour and weekday
    if len(data_copy) >= min_length:
        pivot_data = data_copy.groupby(['weekday', 'hour'])['density'].mean().reset_index()
        pivot_table = pivot_data.pivot(index='weekday', columns='hour', values='density')
        
        # Reorder weekdays
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_table = pivot_table.reindex(weekday_order)
        
        fig = px.imshow(
            pivot_table,
            title="Density Patterns by Day and Hour",
            labels={'x': 'Hour of Day', 'y': 'Day of Week', 'color': 'Avg Density'},
            aspect='auto'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Pattern insights
        st.subheader("Pattern Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Peak hours by weekday
            peak_by_day = pivot_table.idxmax(axis=1)
            st.write("**Peak Hours by Day:**")
            for day, hour in peak_by_day.items():
                if not pd.isna(hour):
                    st.write(f"📅 {day}: {hour}:00")
        
        with col2:
            # Consistent patterns
            hourly_std = pivot_table.std(axis=0)
            consistent_hours = hourly_std.nsmallest(3)
            st.write("**Most Consistent Hours:**")
            for hour, std in consistent_hours.items():
                st.write(f"⏰ {hour}:00 (σ: {std:.2f})")
    else:
        st.info(f"Need at least {min_length} data points for pattern analysis.")

def show_correlation_analysis(data):
    """Show correlation analysis between cameras"""
    st.subheader("Camera Correlation Analysis")
    
    # Create pivot table with cameras as columns
    if len(data['camera_id'].unique()) < 2:
        st.info("Need data from at least 2 cameras for correlation analysis.")
        return
    
    # Resample data to regular intervals for correlation
    correlation_data = []
    
    for camera in data['camera_id'].unique():
        camera_data = data[data['camera_id'] == camera].set_index('timestamp')
        # Resample to 5-minute intervals
        resampled = camera_data['density'].resample('5T').mean()
        correlation_data.append(resampled.rename(camera))
    
    # Combine into single DataFrame
    corr_df = pd.concat(correlation_data, axis=1)
    corr_df = corr_df.dropna()  # Remove rows with missing data
    
    if len(corr_df) < 10:
        st.info("Insufficient overlapping data for correlation analysis.")
        return
    
    # Calculate correlation matrix
    correlation_matrix = corr_df.corr()
    
    # Create correlation heatmap
    fig = px.imshow(
        correlation_matrix,
        title="Camera Density Correlation Matrix",
        color_continuous_scale="RdBu",
        aspect='auto'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Strongest Correlations:**")
        # Get upper triangle of correlation matrix
        mask = np.triu(np.ones_like(correlation_matrix), k=1).astype(bool)
        correlations = correlation_matrix.where(mask).stack().sort_values(ascending=False)
        
        for (cam1, cam2), corr in correlations.head(3).items():
            st.write(f"📹 {cam1} ↔ {cam2}: {corr:.3f}")
    
    with col2:
        st.write("**Weakest Correlations:**")
        for (cam1, cam2), corr in correlations.tail(3).items():
            st.write(f"📹 {cam1} ↔ {cam2}: {corr:.3f}")

def show_clustering_analysis(data):
    """Show clustering analysis of density patterns"""
    st.subheader("Density Pattern Clustering")
    
    # This is a simplified clustering analysis
    # In a full implementation, you might use techniques like K-means or DBSCAN
    
    if len(data) < 20:
        st.info("Need more data points for clustering analysis.")
        return
    
    # Create features for clustering (hour, weekday, density)
    cluster_data = data.copy()
    cluster_data['hour'] = cluster_data['timestamp'].dt.hour
    cluster_data['weekday_num'] = cluster_data['timestamp'].dt.weekday
    
    # Simple clustering based on density ranges
    def categorize_density(density):
        if density < 1.0:
            return "Low"
        elif density < 3.0:
            return "Medium"
        elif density < 5.0:
            return "High"
        else:
            return "Critical"
    
    cluster_data['density_category'] = cluster_data['density'].apply(categorize_density)
    
    # Visualize clusters
    fig = px.scatter_3d(
        cluster_data,
        x='hour',
        y='weekday_num',
        z='density',
        color='density_category',
        title="3D Clustering of Density Patterns",
        labels={
            'hour': 'Hour of Day',
            'weekday_num': 'Day of Week',
            'density': 'Density (people/m²)'
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster summary
    cluster_summary = cluster_data.groupby('density_category').agg({
        'density': ['count', 'mean', 'std'],
        'hour': lambda x: x.mode().iloc[0] if not x.empty else 0
    }).round(2)
    
    st.subheader("Cluster Summary")
    st.dataframe(cluster_summary, use_container_width=True)

def show_statistical_reports(data_manager, alert_system):
    """Show comprehensive statistical reports"""
    st.header("📊 Statistical Reports")
    
    data = get_filtered_data(data_manager)
    
    if data.empty:
        st.warning("No data available for statistical analysis.")
        return
    
    # Report type selection
    report_type = st.selectbox(
        "Select Report Type",
        ["Summary Statistics", "Distribution Analysis", "Performance Metrics", "Alert Statistics"],
        key="report_type"
    )
    
    if report_type == "Summary Statistics":
        show_summary_statistics(data)
    elif report_type == "Distribution Analysis":
        show_distribution_analysis(data)
    elif report_type == "Performance Metrics":
        show_performance_metrics(data_manager)
    else:  # Alert Statistics
        show_alert_statistics(data_manager, alert_system)

def show_summary_statistics(data):
    """Show comprehensive summary statistics"""
    st.subheader("Summary Statistics")
    
    # Overall statistics
    overall_stats = data['density'].describe()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(data))
        st.metric("Mean Density", f"{overall_stats['mean']:.2f}")
    
    with col2:
        st.metric("Median Density", f"{overall_stats['50%']:.2f}")
        st.metric("Std Deviation", f"{overall_stats['std']:.2f}")
    
    with col3:
        st.metric("Min Density", f"{overall_stats['min']:.2f}")
        st.metric("Max Density", f"{overall_stats['max']:.2f}")
    
    with col4:
        st.metric("25th Percentile", f"{overall_stats['25%']:.2f}")
        st.metric("75th Percentile", f"{overall_stats['75%']:.2f}")
    
    # Statistics by camera
    st.subheader("Statistics by Camera")
    
    camera_stats = data.groupby('camera_id')['density'].describe().round(2)
    st.dataframe(camera_stats, use_container_width=True)
    
    # Time-based statistics
    st.subheader("Time-based Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hourly statistics
        data_copy = data.copy()
        data_copy['hour'] = data_copy['timestamp'].dt.hour
        hourly_stats = data_copy.groupby('hour')['density'].agg(['mean', 'std', 'count']).round(2)
        
        fig = px.bar(
            x=hourly_stats.index,
            y=hourly_stats['mean'],
            error_y=hourly_stats['std'],
            title="Average Density by Hour",
            labels={'x': 'Hour', 'y': 'Average Density'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Daily statistics
        data_copy['weekday'] = data_copy['timestamp'].dt.day_name()
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_stats = data_copy.groupby('weekday')['density'].mean().reindex(weekday_order)
        
        fig = px.bar(
            x=daily_stats.index,
            y=daily_stats.values,
            title="Average Density by Day of Week",
            labels={'x': 'Day', 'y': 'Average Density'}
        )
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

def show_distribution_analysis(data):
    """Show distribution analysis"""
    st.subheader("Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig = px.histogram(
            data,
            x='density',
            nbins=30,
            title="Density Distribution",
            labels={'density': 'Density (people/m²)', 'count': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution by camera
        fig = px.box(
            data,
            x='camera_id',
            y='density',
            title="Density Distribution by Camera"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Q-Q plot (simplified)
        st.write("**Distribution Characteristics:**")
        
        density_data = data['density']
        
        # Calculate skewness and kurtosis
        skewness = stats.skew(density_data)
        kurt = stats.kurtosis(density_data)
        
        st.write(f"📊 **Skewness:** {skewness:.3f}")
        if abs(skewness) < 0.5:
            st.write("   → Nearly symmetric distribution")
        elif skewness > 0:
            st.write("   → Right-skewed (tail extends right)")
        else:
            st.write("   → Left-skewed (tail extends left)")
        
        st.write(f"📈 **Kurtosis:** {kurt:.3f}")
        if abs(kurt) < 0.5:
            st.write("   → Normal tail behavior")
        elif kurt > 0:
            st.write("   → Heavy-tailed distribution")
        else:
            st.write("   → Light-tailed distribution")
        
        # Normality test (Shapiro-Wilk for small samples)
        if len(density_data) <= 5000:
            stat, p_value = stats.shapiro(density_data)
            st.write(f"🔬 **Normality Test (p-value):** {p_value:.6f}")
            if p_value > 0.05:
                st.write("   → Data appears normally distributed")
            else:
                st.write("   → Data does not appear normally distributed")
        
        # Percentile analysis
        st.write("**Percentile Analysis:**")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        perc_values = np.percentile(density_data, percentiles)
        
        for p, v in zip(percentiles, perc_values):
            st.write(f"   {p}th percentile: {v:.2f}")

def show_performance_metrics(data_manager):
    """Show system performance metrics"""
    st.subheader("System Performance Metrics")
    
    # Get performance data
    camera_performance = data_manager.get_camera_performance()
    db_info = data_manager.get_database_info()
    
    # Performance overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Active Cameras", len(camera_performance))
        total_readings = sum(stats.get('total_readings', 0) for stats in camera_performance.values())
        st.metric("Total Readings", total_readings)
    
    with col2:
        avg_uptime = np.mean([stats.get('uptime_percentage', 0) for stats in camera_performance.values()])
        st.metric("Average Uptime", f"{avg_uptime:.1f}%")
        
        db_size_mb = db_info.get('database_size_bytes', 0) / (1024 * 1024)
        st.metric("Database Size", f"{db_size_mb:.1f} MB")
    
    with col3:
        total_alerts = sum(stats.get('alert_count', 0) for stats in camera_performance.values())
        st.metric("Total Alerts", total_alerts)
        
        cache_size = db_info.get('cache_size', 0)
        st.metric("Cache Size", cache_size)
    
    # Camera performance table
    if camera_performance:
        st.subheader("Camera Performance Details")
        
        perf_df = pd.DataFrame(camera_performance).T
        perf_df = perf_df.round(2)
        st.dataframe(perf_df, use_container_width=True)

def show_alert_statistics(data_manager, alert_system):
    """Show detailed alert statistics"""
    st.subheader("Alert Statistics")
    
    # Get alert data
    alert_summary = alert_system.get_alert_summary(hours=168)  # Last week
    alert_events = data_manager.get_alert_events(hours=168)
    
    # Alert overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Alerts (7d)", alert_summary.get('total_alerts', 0))
    
    with col2:
        warning_count = alert_summary.get('alerts_by_level', {}).get('Warning', 0)
        st.metric("Warning Alerts", warning_count)
    
    with col3:
        critical_count = alert_summary.get('alerts_by_level', {}).get('Critical', 0)
        st.metric("Critical Alerts", critical_count)
    
    with col4:
        avg_response = alert_summary.get('average_response_time_seconds', 0)
        st.metric("Avg Response Time", f"{avg_response:.1f}s")
    
    if not alert_events.empty:
        # Alert timeline
        st.subheader("Alert Timeline")
        
        fig = px.histogram(
            alert_events,
            x='timestamp',
            color='alert_level',
            title="Alert Frequency Over Time",
            nbins=50
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Alert statistics table
        st.subheader("Alert Details")
        
        alert_stats = alert_events.groupby(['camera_id', 'alert_level']).size().reset_index(name='count')
        alert_pivot = alert_stats.pivot(index='camera_id', columns='alert_level', values='count').fillna(0)
        st.dataframe(alert_pivot, use_container_width=True)

def show_predictive_analytics(data_manager):
    """Show predictive analytics and forecasting"""
    st.header("🎯 Predictive Analytics")
    
    data = get_filtered_data(data_manager)
    
    if data.empty or len(data) < 10:
        st.warning("Insufficient data for predictive analytics. Need at least 10 data points.")
        return
    
    # Prediction options
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Prediction Settings")
        
        prediction_type = st.selectbox(
            "Prediction Type",
            ["Short-term Forecast", "Peak Time Prediction", "Alert Probability", "Capacity Planning"],
            key="prediction_type"
        )
        
        forecast_horizon = st.slider(
            "Forecast Horizon (hours)",
            1, 24, 6,
            help="How far ahead to predict"
        )
        
        selected_camera = st.selectbox(
            "Camera for Prediction",
            data['camera_id'].unique(),
            key="pred_camera"
        )
    
    with col1:
        camera_data = data[data['camera_id'] == selected_camera].sort_values('timestamp')
        
        if len(camera_data) < 5:
            st.warning(f"Insufficient data for {selected_camera}. Need at least 5 data points.")
            return
        
        if prediction_type == "Short-term Forecast":
            show_short_term_forecast(camera_data, forecast_horizon, selected_camera)
        elif prediction_type == "Peak Time Prediction":
            show_peak_time_prediction(camera_data, selected_camera)
        elif prediction_type == "Alert Probability":
            show_alert_probability(camera_data, selected_camera)
        else:  # Capacity Planning
            show_capacity_planning(camera_data, selected_camera)

def show_short_term_forecast(camera_data, horizon_hours, camera_id):
    """Show short-term density forecast"""
    st.subheader(f"Short-term Forecast for {camera_id}")
    
    if len(camera_data) < 3:
        st.warning("Need more historical data for forecasting.")
        return
    
    # Simple linear extrapolation for demonstration
    # In production, you might use ARIMA, Prophet, or ML models
    
    # Use last few points for trend
    recent_points = min(10, len(camera_data))
    recent_data = camera_data.tail(recent_points).copy()
    
    # Convert timestamps to numeric for regression
    x_numeric = pd.to_numeric(recent_data['timestamp'])
    y = recent_data['density']
    
    # Fit linear model
    if len(recent_data) >= 2:
        slope, intercept, r_value, _, _ = stats.linregr(x_numeric, y)
        
        # Generate future timestamps
        last_timestamp = recent_data['timestamp'].iloc[-1]
        future_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(minutes=5),
            periods=horizon_hours * 12,  # 5-minute intervals
            freq='5T'
        )
        
        # Predict future values
        future_numeric = pd.to_numeric(future_timestamps)
        future_predictions = slope * future_numeric + intercept
        
        # Ensure non-negative predictions
        future_predictions = np.maximum(future_predictions, 0)
        
        # Create forecast plot
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=camera_data['timestamp'],
            y=camera_data['density'],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='blue')
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=future_timestamps,
            y=future_predictions,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        # Add confidence bands (simplified)
        prediction_std = y.std()
        upper_bound = future_predictions + 1.96 * prediction_std
        lower_bound = np.maximum(future_predictions - 1.96 * prediction_std, 0)
        
        fig.add_trace(go.Scatter(
            x=future_timestamps,
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=future_timestamps,
            y=lower_bound,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.1)',
            name='Confidence Band',
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=f"Density Forecast for {camera_id} ({horizon_hours}h ahead)",
            xaxis_title="Time",
            yaxis_title="Density (people/m²)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast summary
        st.subheader("Forecast Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model R²", f"{r_value**2:.3f}")
        
        with col2:
            avg_forecast = np.mean(future_predictions)
            st.metric("Avg Forecast Density", f"{avg_forecast:.2f}")
        
        with col3:
            max_forecast = np.max(future_predictions)
            st.metric("Peak Forecast Density", f"{max_forecast:.2f}")
        
        # Alert predictions
        warning_threshold = 2.0  # Default threshold
        critical_threshold = 4.0
        
        warning_periods = np.sum(future_predictions >= warning_threshold)
        critical_periods = np.sum(future_predictions >= critical_threshold)
        
        if warning_periods > 0 or critical_periods > 0:
            st.warning(f"⚠️ Predicted Alert Periods:")
            if critical_periods > 0:
                st.error(f"🔴 Critical alerts expected in {critical_periods} periods")
            if warning_periods > 0:
                st.warning(f"🟡 Warning alerts expected in {warning_periods} periods")

def show_peak_time_prediction(camera_data, camera_id):
    """Show peak time prediction based on historical patterns"""
    st.subheader(f"Peak Time Prediction for {camera_id}")
    
    # Analyze historical peaks
    camera_data = camera_data.copy()
    camera_data['hour'] = camera_data['timestamp'].dt.hour
    camera_data['weekday'] = camera_data['timestamp'].dt.day_name()
    
    # Find daily patterns
    hourly_pattern = camera_data.groupby('hour')['density'].agg(['mean', 'std', 'max'])
    
    # Predict next peak
    current_time = datetime.now()
    current_hour = current_time.hour
    
    # Find next hours with high density
    future_hours = [(current_hour + i) % 24 for i in range(1, 25)]
    peak_predictions = []
    
    for hour in future_hours:
        if hour in hourly_pattern.index:
            expected_density = hourly_pattern.loc[hour, 'mean']
            confidence = 1 / (1 + hourly_pattern.loc[hour, 'std'])  # Inverse of std as confidence
            peak_predictions.append({
                'hour': hour,
                'expected_density': expected_density,
                'confidence': confidence
            })
    
    if peak_predictions:
        predictions_df = pd.DataFrame(peak_predictions)
        
        # Find top predicted peaks
        top_peaks = predictions_df.nlargest(5, 'expected_density')
        
        st.write("**Next Predicted Peaks (24h):**")
        for _, peak in top_peaks.iterrows():
            confidence_level = "High" if peak['confidence'] > 0.7 else "Medium" if peak['confidence'] > 0.4 else "Low"
            st.write(f"⏰ {peak['hour']:02d}:00 - Expected: {peak['expected_density']:.2f} (Confidence: {confidence_level})")
    
    # Visualize hourly pattern
    fig = px.bar(
        x=hourly_pattern.index,
        y=hourly_pattern['mean'],
        error_y=hourly_pattern['std'],
        title=f"Average Density Pattern by Hour - {camera_id}",
        labels={'x': 'Hour of Day', 'y': 'Average Density'}
    )
    
    # Highlight current hour
    fig.add_vline(x=current_time.hour, line_dash="dash", line_color="red", 
                  annotation_text="Current Hour")
    
    st.plotly_chart(fig, use_container_width=True)

def show_alert_probability(camera_data, camera_id):
    """Show probability of alerts based on historical data"""
    st.subheader(f"Alert Probability Analysis for {camera_id}")
    
    # Calculate historical alert rates
    warning_threshold = 2.0
    critical_threshold = 4.0
    
    total_readings = len(camera_data)
    warning_count = len(camera_data[camera_data['density'] >= warning_threshold])
    critical_count = len(camera_data[camera_data['density'] >= critical_threshold])
    
    warning_probability = warning_count / total_readings if total_readings > 0 else 0
    critical_probability = critical_count / total_readings if total_readings > 0 else 0
    
    # Show probabilities
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Warning Alert Probability", f"{warning_probability:.1%}")
    
    with col2:
        st.metric("Critical Alert Probability", f"{critical_probability:.1%}")
    
    with col3:
        st.metric("Normal Probability", f"{(1 - warning_probability):.1%}")
    
    # Risk assessment by time periods
    camera_data = camera_data.copy()
    camera_data['hour'] = camera_data['timestamp'].dt.hour
    camera_data['is_warning'] = camera_data['density'] >= warning_threshold
    camera_data['is_critical'] = camera_data['density'] >= critical_threshold
    
    # Calculate hourly risk
    hourly_risk = camera_data.groupby('hour').agg({
        'is_warning': 'mean',
        'is_critical': 'mean',
        'density': 'count'
    }).rename(columns={'density': 'readings_count'})
    
    # Only show hours with sufficient data
    hourly_risk = hourly_risk[hourly_risk['readings_count'] >= 3]
    
    if not hourly_risk.empty:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=hourly_risk.index,
            y=hourly_risk['is_warning'] * 100,
            name='Warning Probability',
            marker_color='orange'
        ))
        
        fig.add_trace(go.Bar(
            x=hourly_risk.index,
            y=hourly_risk['is_critical'] * 100,
            name='Critical Probability',
            marker_color='red'
        ))
        
        fig.update_layout(
            title=f"Alert Probability by Hour - {camera_id}",
            xaxis_title="Hour of Day",
            yaxis_title="Probability (%)",
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk recommendations
        high_risk_hours = hourly_risk[hourly_risk['is_critical'] > 0.1].index.tolist()
        if high_risk_hours:
            st.warning(f"⚠️ High Risk Hours: {', '.join([f'{h:02d}:00' for h in high_risk_hours])}")

def show_capacity_planning(camera_data, camera_id):
    """Show capacity planning analysis"""
    st.subheader(f"Capacity Planning for {camera_id}")
    
    # Calculate capacity metrics
    max_observed = camera_data['density'].max()
    avg_density = camera_data['density'].mean()
    p95_density = np.percentile(camera_data['density'], 95)
    p99_density = np.percentile(camera_data['density'], 99)
    
    # Capacity recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Current Capacity Metrics:**")
        st.metric("Average Density", f"{avg_density:.2f} people/m²")
        st.metric("95th Percentile", f"{p95_density:.2f} people/m²")
        st.metric("99th Percentile", f"{p99_density:.2f} people/m²")
        st.metric("Maximum Observed", f"{max_observed:.2f} people/m²")
    
    with col2:
        st.write("**Capacity Recommendations:**")
        
        # Safety margins
        recommended_warning = p95_density * 0.8
        recommended_critical = p95_density * 0.9
        
        st.write(f"🟡 **Recommended Warning Threshold:** {recommended_warning:.2f} people/m²")
        st.write(f"🔴 **Recommended Critical Threshold:** {recommended_critical:.2f} people/m²")
        
        # Utilization analysis
        warning_threshold = 2.0
        utilization = (avg_density / warning_threshold) * 100
        st.write(f"📊 **Current Utilization:** {utilization:.1f}%")
        
        if utilization > 80:
            st.error("⚠️ High utilization - consider capacity expansion")
        elif utilization > 60:
            st.warning("⚠️ Moderate utilization - monitor closely")
        else:
            st.success("✅ Good capacity headroom")
    
    # Capacity over time
    camera_data = camera_data.copy()
    camera_data['utilization'] = (camera_data['density'] / 4.0) * 100  # Assuming 4.0 as max capacity
    
    fig = px.line(
        camera_data,
        x='timestamp',
        y='utilization',
        title=f"Capacity Utilization Over Time - {camera_id}",
        labels={'utilization': 'Utilization (%)'}
    )
    
    # Add capacity lines
    fig.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="80% Capacity")
    fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="100% Capacity")
    
    st.plotly_chart(fig, use_container_width=True)

def show_custom_reports(data_manager):
    """Show custom report generation"""
    st.header("📋 Custom Reports")
    
    st.subheader("Report Generator")
    
    # Report configuration
    col1, col2 = st.columns(2)
    
    with col1:
        report_name = st.text_input("Report Name", value="Custom Analysis Report")
        
        report_type = st.selectbox(
            "Report Type",
            ["Executive Summary", "Technical Analysis", "Compliance Report", "Performance Review"]
        )
        
        include_charts = st.checkbox("Include Charts", value=True)
        include_raw_data = st.checkbox("Include Raw Data", value=False)
    
    with col2:
        report_period = st.selectbox(
            "Report Period",
            ["Last 24 Hours", "Last Week", "Last Month", "Custom Period"]
        )
        
        if report_period == "Custom Period":
            start_date = st.date_input("Start Date")
            end_date = st.date_input("End Date")
        
        report_format = st.selectbox(
            "Export Format",
            ["PDF Report", "CSV Data", "JSON Data", "Excel Workbook"]
        )
    
    # Report sections
    st.subheader("Report Sections")
    
    sections = {
        "Executive Summary": st.checkbox("Executive Summary", value=True),
        "Key Metrics": st.checkbox("Key Metrics", value=True),
        "Trend Analysis": st.checkbox("Trend Analysis", value=True),
        "Alert Summary": st.checkbox("Alert Summary", value=True),
        "Camera Performance": st.checkbox("Camera Performance", value=True),
        "Recommendations": st.checkbox("Recommendations", value=False)
    }
    
    # Generate report button
    if st.button("Generate Custom Report", type="primary"):
        # Get data based on period
        hours_map = {
            "Last 24 Hours": 24,
            "Last Week": 168,
            "Last Month": 720
        }
        
        hours = hours_map.get(report_period, 24)
        data = data_manager.get_recent_data(hours=hours)
        
        if data.empty:
            st.error("No data available for the selected period.")
            return
        
        # Generate report content
        report_content = generate_report_content(data, sections, data_manager)
        
        # Display report preview
        st.subheader("Report Preview")
        st.markdown(report_content)
        
        # Download options
        if report_format == "CSV Data":
            csv_data = data.to_csv(index=False)
            st.download_button(
                "Download CSV Report",
                csv_data,
                file_name=f"{report_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        elif report_format == "JSON Data":
            json_data = data.to_json(orient='records', date_format='iso', indent=2)
            st.download_button(
                "Download JSON Report",
                json_data,
                file_name=f"{report_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )

def generate_report_content(data, sections, data_manager):
    """Generate custom report content"""
    report_lines = []
    
    report_lines.append(f"# Custom Analysis Report")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Data Period:** {data['timestamp'].min().strftime('%Y-%m-%d')} to {data['timestamp'].max().strftime('%Y-%m-%d')}")
    report_lines.append(f"**Total Records:** {len(data)}")
    report_lines.append("")
    
    if sections.get("Executive Summary"):
        report_lines.extend(generate_executive_summary(data))
    
    if sections.get("Key Metrics"):
        report_lines.extend(generate_key_metrics(data))
    
    if sections.get("Trend Analysis"):
        report_lines.extend(generate_trend_summary(data))
    
    if sections.get("Alert Summary"):
        report_lines.extend(generate_alert_summary(data))
    
    if sections.get("Camera Performance"):
        report_lines.extend(generate_camera_performance_summary(data_manager))
    
    if sections.get("Recommendations"):
        report_lines.extend(generate_recommendations(data))
    
    return "\n".join(report_lines)

def generate_executive_summary(data):
    """Generate executive summary section"""
    lines = ["## Executive Summary", ""]
    
    # Key findings
    avg_density = data['density'].mean()
    max_density = data['density'].max()
    alert_count = len(data[data['alert_level'] != 'Normal'])
    
    lines.append(f"- **Average Density:** {avg_density:.2f} people/m²")
    lines.append(f"- **Peak Density:** {max_density:.2f} people/m²")
    lines.append(f"- **Total Alerts:** {alert_count}")
    lines.append(f"- **Active Cameras:** {len(data['camera_id'].unique())}")
    lines.append("")
    
    return lines

def generate_key_metrics(data):
    """Generate key metrics section"""
    lines = ["## Key Metrics", ""]
    
    stats = data['density'].describe()
    
    lines.append("### Density Statistics")
    lines.append(f"- **Mean:** {stats['mean']:.2f} people/m²")
    lines.append(f"- **Median:** {stats['50%']:.2f} people/m²")
    lines.append(f"- **Standard Deviation:** {stats['std']:.2f}")
    lines.append(f"- **Range:** {stats['min']:.2f} - {stats['max']:.2f} people/m²")
    lines.append("")
    
    return lines

def generate_trend_summary(data):
    """Generate trend analysis summary"""
    lines = ["## Trend Analysis", ""]
    
    # Simple trend analysis
    for camera in data['camera_id'].unique():
        camera_data = data[data['camera_id'] == camera].sort_values('timestamp')
        if len(camera_data) >= 2:
            first_half = camera_data.head(len(camera_data)//2)['density'].mean()
            second_half = camera_data.tail(len(camera_data)//2)['density'].mean()
            trend = "increasing" if second_half > first_half else "decreasing"
            change = abs(second_half - first_half)
            
            lines.append(f"- **{camera}:** {trend} trend (±{change:.2f} people/m²)")
    
    lines.append("")
    return lines

def generate_alert_summary(data):
    """Generate alert summary section"""
    lines = ["## Alert Summary", ""]
    
    alert_counts = data['alert_level'].value_counts()
    
    for level, count in alert_counts.items():
        percentage = (count / len(data)) * 100
        lines.append(f"- **{level}:** {count} alerts ({percentage:.1f}%)")
    
    lines.append("")
    return lines

def generate_camera_performance_summary(data_manager):
    """Generate camera performance summary"""
    lines = ["## Camera Performance", ""]
    
    camera_performance = data_manager.get_camera_performance()
    
    for camera, stats in camera_performance.items():
        uptime = stats.get('uptime_percentage', 0)
        avg_density = stats.get('average_density', 0)
        alert_count = stats.get('alert_count', 0)
        
        lines.append(f"### {camera}")
        lines.append(f"- **Uptime:** {uptime:.1f}%")
        lines.append(f"- **Average Density:** {avg_density:.2f} people/m²")
        lines.append(f"- **Alert Count:** {alert_count}")
        lines.append("")
    
    return lines

def generate_recommendations(data):
    """Generate recommendations section"""
    lines = ["## Recommendations", ""]
    
    avg_density = data['density'].mean()
    max_density = data['density'].max()
    
    if max_density > 4.0:
        lines.append("- ⚠️ **High density levels observed.** Consider increasing monitoring frequency during peak hours.")
    
    if avg_density > 2.0:
        lines.append("- 📊 **Above-average density levels.** Review capacity planning and crowd management procedures.")
    
    alert_rate = len(data[data['alert_level'] != 'Normal']) / len(data)
    if alert_rate > 0.1:
        lines.append("- 🚨 **High alert frequency.** Consider adjusting alert thresholds or improving crowd flow management.")
    
    lines.append("- 📈 **Continue monitoring trends** and adjust thresholds based on operational requirements.")
    lines.append("")
    
    return lines
