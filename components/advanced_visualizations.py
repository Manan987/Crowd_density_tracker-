import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import cv2

class AdvancedVisualizationEngine:
    """Advanced visualization components for crowd density monitoring"""
    
    def __init__(self):
        self.color_scales = {
            'density': 'Viridis',
            'heat': 'Hot', 
            'safety': 'RdYlGn_r'
        }
    
    def create_3d_density_surface(self, density_data, camera_id="CAM_001"):
        """Create 3D surface plot of crowd density"""
        # Generate synthetic 3D density data for demonstration
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 100, 50)
        X, Y = np.meshgrid(x, y)
        
        # Create realistic density distribution
        center_x, center_y = 50, 50
        Z = np.exp(-((X - center_x)**2 + (Y - center_y)**2) / 800)
        
        # Add some noise and hotspots
        for _ in range(3):
            hx, hy = np.random.randint(20, 80, 2)
            intensity = np.random.uniform(0.3, 0.8)
            Z += intensity * np.exp(-((X - hx)**2 + (Y - hy)**2) / 400)
        
        # Scale to realistic density values
        Z = Z * 4.0  # Max 4 people/m²
        
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            colorbar=dict(
                title="Density (people/m²)",
                titleside="right",
                tickmode="linear",
                tick0=0,
                dtick=1
            ),
            hovertemplate='X: %{x}<br>Y: %{y}<br>Density: %{z:.2f}<extra></extra>'
        )])
        
        fig.update_layout(
            title=f'3D Crowd Density Surface - {camera_id}',
            scene=dict(
                xaxis_title='X Position (meters)',
                yaxis_title='Y Position (meters)',
                zaxis_title='Density (people/m²)',
                camera=dict(eye=dict(x=1.2, y=1.2, z=0.8)),
                bgcolor='rgba(0,0,0,0)'
            ),
            height=500,
            margin=dict(l=0, r=0, t=50, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def create_realtime_heatmap(self, frame_shape=(480, 640), density_points=None):
        """Create real-time density heatmap"""
        if density_points is None:
            # Generate realistic density hotspots
            density_points = []
            for _ in range(np.random.randint(3, 8)):
                x = np.random.randint(50, frame_shape[1] - 50)
                y = np.random.randint(50, frame_shape[0] - 50)
                intensity = np.random.uniform(0.5, 3.0)
                radius = np.random.randint(30, 80)
                density_points.append({'x': x, 'y': y, 'intensity': intensity, 'radius': radius})
        
        # Create coordinate grids
        x = np.arange(0, frame_shape[1], 10)
        y = np.arange(0, frame_shape[0], 10)
        X, Y = np.meshgrid(x, y)
        
        # Generate density map
        density_map = np.zeros_like(X, dtype=float)
        
        for point in density_points:
            distance = np.sqrt((X - point['x'])**2 + (Y - point['y'])**2)
            density_contribution = point['intensity'] * np.exp(-distance / point['radius'])
            density_map += density_contribution
        
        fig = go.Figure(data=go.Heatmap(
            x=x,
            y=y,
            z=density_map,
            colorscale='Hot',
            hoverongaps=False,
            colorbar=dict(
                title="Density Level",
                titleside="right"
            )
        ))
        
        fig.update_layout(
            title='Real-time Crowd Density Heatmap',
            xaxis_title='X Position (pixels)',
            yaxis_title='Y Position (pixels)',
            height=400,
            yaxis=dict(autorange='reversed'),  # Flip Y-axis to match image coordinates
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def create_flow_vectors(self, camera_id="CAM_001"):
        """Create crowd movement flow visualization"""
        # Generate sample movement vectors
        x = np.arange(10, 100, 15)
        y = np.arange(10, 80, 15)
        X, Y = np.meshgrid(x, y)
        
        # Create flow field (people moving towards exits)
        center_x, center_y = 50, 40
        U = (X - center_x) / 30 + np.random.normal(0, 0.3, X.shape)
        V = (Y - center_y) / 30 + np.random.normal(0, 0.3, Y.shape)
        
        # Normalize vectors
        magnitude = np.sqrt(U**2 + V**2)
        U = U / (magnitude + 1e-8) * np.clip(magnitude, 0, 2)
        V = V / (magnitude + 1e-8) * np.clip(magnitude, 0, 2)
        
        fig = go.Figure()
        
        # Add vector field
        for i in range(len(x)):
            for j in range(len(y)):
                fig.add_trace(go.Scatter(
                    x=[X[j,i], X[j,i] + U[j,i]*5],
                    y=[Y[j,i], Y[j,i] + V[j,i]*5],
                    mode='lines',
                    line=dict(color='cyan', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Add arrowheads
                fig.add_trace(go.Scatter(
                    x=[X[j,i] + U[j,i]*5],
                    y=[Y[j,i] + V[j,i]*5],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=8,
                        color='cyan',
                        angle=np.degrees(np.arctan2(V[j,i], U[j,i]))
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        fig.update_layout(
            title=f'Crowd Movement Flow - {camera_id}',
            xaxis_title='X Position (meters)',
            yaxis_title='Y Position (meters)',
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.2)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.2)')
        )
        
        return fig
    
    def create_risk_assessment_gauge(self, current_density, camera_id="CAM_001"):
        """Create risk assessment gauge chart"""
        # Calculate risk score based on density
        if current_density < 1.0:
            risk_score = current_density * 25
            risk_level = "LOW"
            color = "green"
        elif current_density < 2.5:
            risk_score = 25 + (current_density - 1.0) * 33.33
            risk_level = "MODERATE"
            color = "yellow"
        elif current_density < 4.0:
            risk_score = 75 + (current_density - 2.5) * 16.67
            risk_level = "HIGH"
            color = "red"
        else:
            risk_score = 100
            risk_level = "CRITICAL"
            color = "darkred"
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Risk Assessment - {camera_id}"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': "white"},
            annotations=[
                dict(
                    x=0.5, y=0.1,
                    text=f"Risk Level: {risk_level}",
                    showarrow=False,
                    font=dict(size=16, color=color)
                )
            ]
        )
        
        return fig

def show_advanced_monitoring_dashboard():
    """Display advanced monitoring dashboard with 3D visualizations"""
    viz_engine = AdvancedVisualizationEngine()
    
    st.markdown("## Advanced Monitoring Dashboard")
    
    # Top row - 3D Surface and Risk Assessment
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 3D Density Surface")
        density_data = st.session_state.data_manager.get_recent_data(minutes=5)
        fig_3d = viz_engine.create_3d_density_surface(density_data)
        st.plotly_chart(fig_3d, use_container_width=True)
    
    with col2:
        st.markdown("### Risk Assessment")
        current_density = st.session_state.get('current_density', 0)
        risk_gauge = viz_engine.create_risk_assessment_gauge(current_density)
        st.plotly_chart(risk_gauge, use_container_width=True)
    
    # Middle row - Heatmap and Flow Vectors
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Real-time Heatmap")
        heatmap = viz_engine.create_realtime_heatmap()
        st.plotly_chart(heatmap, use_container_width=True)
    
    with col2:
        st.markdown("### Movement Flow")
        flow_chart = viz_engine.create_flow_vectors()
        st.plotly_chart(flow_chart, use_container_width=True)