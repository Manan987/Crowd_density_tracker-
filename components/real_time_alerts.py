import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time

class RealTimeAlertSystem:
    """Real-time alert notification and management system"""
    
    def __init__(self):
        self.alert_history = []
        self.active_alerts = {}
        self.notification_queue = []
    
    def show_live_alert_dashboard(self):
        """Display comprehensive live alert dashboard"""
        st.markdown("## 🚨 Live Alert Management Center")
        
        # Alert overview cards
        self.show_alert_overview_cards()
        
        # Main alert interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.show_alert_map_interface()
            self.show_alert_timeline()
        
        with col2:
            self.show_active_alerts_panel()
            self.show_notification_center()
    
    def show_alert_overview_cards(self):
        """Display alert overview metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        # Get recent alert data
        alert_data = st.session_state.data_manager.get_alert_events(hours=1)
        
        with col1:
            total_alerts = len(alert_data) if not alert_data.empty else 0
            st.markdown(f"""
            <div class="metric-card">
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; color: #ff6b6b;">🚨</div>
                    <div style="font-size: 2rem; font-weight: bold; color: white;">{total_alerts}</div>
                    <div style="color: rgba(255,255,255,0.8);">Total Alerts (1h)</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            critical_alerts = len(alert_data[alert_data['alert_level'] == 'Critical']) if not alert_data.empty else 0
            st.markdown(f"""
            <div class="metric-card">
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; color: #ff4757;">🔴</div>
                    <div style="font-size: 2rem; font-weight: bold; color: white;">{critical_alerts}</div>
                    <div style="color: rgba(255,255,255,0.8);">Critical Alerts</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_response = np.random.uniform(30, 120) if total_alerts > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; color: #3742fa;">⏱️</div>
                    <div style="font-size: 2rem; font-weight: bold; color: white;">{avg_response:.0f}s</div>
                    <div style="color: rgba(255,255,255,0.8);">Avg Response</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            system_status = "OPERATIONAL"
            status_color = "#2ed573"
            st.markdown(f"""
            <div class="metric-card">
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; color: {status_color};">✅</div>
                    <div style="font-size: 1.2rem; font-weight: bold; color: white;">{system_status}</div>
                    <div style="color: rgba(255,255,255,0.8);">System Status</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def show_alert_map_interface(self):
        """Interactive alert map showing camera locations and alert status"""
        st.markdown("### 🗺️ Camera Alert Map")
        
        # Create interactive map using plotly
        camera_locations = {
            'CAM_001': {'x': 20, 'y': 80, 'name': 'Main Entrance', 'status': 'normal'},
            'CAM_002': {'x': 50, 'y': 50, 'name': 'Central Plaza', 'status': 'warning'},
            'CAM_003': {'x': 80, 'y': 80, 'name': 'Exit Gate', 'status': 'normal'},
            'CAM_004': {'x': 50, 'y': 20, 'name': 'Emergency Route', 'status': 'critical'}
        }
        
        fig = go.Figure()
        
        # Add venue layout background
        fig.add_shape(
            type="rect",
            x0=10, y0=10, x1=90, y1=90,
            line=dict(color="rgba(255,255,255,0.3)", width=2),
            fillcolor="rgba(255,255,255,0.1)"
        )
        
        # Add camera locations with status indicators
        for cam_id, info in camera_locations.items():
            color_map = {
                'normal': '#2ed573',
                'warning': '#ffa726', 
                'critical': '#ff4757'
            }
            
            size_map = {
                'normal': 15,
                'warning': 20,
                'critical': 25
            }
            
            fig.add_trace(go.Scatter(
                x=[info['x']],
                y=[info['y']],
                mode='markers+text',
                marker=dict(
                    size=size_map[info['status']],
                    color=color_map[info['status']],
                    symbol='circle',
                    line=dict(width=3, color='white')
                ),
                text=[cam_id],
                textposition="bottom center",
                textfont=dict(color="white", size=12),
                name=f"{cam_id} - {info['name']}",
                hovertemplate=f"<b>{cam_id}</b><br>{info['name']}<br>Status: {info['status'].title()}<extra></extra>"
            ))
        
        # Add alert zones
        if camera_locations['CAM_002']['status'] == 'warning':
            fig.add_shape(
                type="circle",
                x0=45, y0=45, x1=55, y1=55,
                line=dict(color="rgba(255,167,38,0.5)", width=2),
                fillcolor="rgba(255,167,38,0.2)"
            )
        
        if camera_locations['CAM_004']['status'] == 'critical':
            fig.add_shape(
                type="circle",
                x0=40, y0=10, x1=60, y1=30,
                line=dict(color="rgba(255,71,87,0.7)", width=3),
                fillcolor="rgba(255,71,87,0.3)"
            )
        
        fig.update_layout(
            title="Live Camera Status Map",
            xaxis=dict(range=[0, 100], showgrid=False, showticklabels=False),
            yaxis=dict(range=[0, 100], showgrid=False, showticklabels=False),
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_alert_timeline(self):
        """Real-time alert timeline visualization"""
        st.markdown("### ⏰ Alert Timeline")
        
        # Get recent alert events
        alert_events = st.session_state.data_manager.get_alert_events(hours=2)
        
        if not alert_events.empty:
            fig = go.Figure()
            
            # Color mapping for alert levels
            color_map = {
                'Normal': '#2ed573',
                'Warning': '#ffa726',
                'Critical': '#ff4757'
            }
            
            for alert_level in alert_events['alert_level'].unique():
                level_data = alert_events[alert_events['alert_level'] == alert_level]
                
                fig.add_trace(go.Scatter(
                    x=level_data['timestamp'],
                    y=level_data['camera_id'],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=color_map.get(alert_level, '#gray'),
                        symbol='circle',
                        line=dict(width=2, color='white')
                    ),
                    name=alert_level,
                    text=level_data['density'].round(1),
                    hovertemplate='<b>%{y}</b><br>Time: %{x}<br>Level: ' + alert_level + '<br>Density: %{text} people/m²<extra></extra>'
                ))
            
            fig.update_layout(
                title="Alert Events Timeline",
                xaxis_title="Time",
                yaxis_title="Camera ID",
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No recent alert events to display")
    
    def show_active_alerts_panel(self):
        """Display active alerts requiring attention"""
        st.markdown("### 🔥 Active Alerts")
        
        # Simulate active alerts for demonstration
        active_alerts = [
            {
                'id': 'ALT_001',
                'camera': 'CAM_004',
                'level': 'Critical',
                'density': 4.2,
                'time': datetime.now() - timedelta(minutes=2),
                'location': 'Emergency Route'
            },
            {
                'id': 'ALT_002', 
                'camera': 'CAM_002',
                'level': 'Warning',
                'density': 2.8,
                'time': datetime.now() - timedelta(minutes=5),
                'location': 'Central Plaza'
            }
        ]
        
        for alert in active_alerts:
            level_colors = {
                'Critical': '#ff4757',
                'Warning': '#ffa726',
                'Normal': '#2ed573'
            }
            
            color = level_colors.get(alert['level'], '#gray')
            time_elapsed = (datetime.now() - alert['time']).seconds // 60
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color}20, {color}10); 
                        border-left: 4px solid {color}; border-radius: 8px; 
                        padding: 1rem; margin: 0.5rem 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="color: {color};">{alert['level'].upper()}</strong>
                        <br><strong>{alert['camera']}</strong> - {alert['location']}
                        <br><small>Density: {alert['density']} people/m² | {time_elapsed}m ago</small>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.5rem;">{'🔴' if alert['level'] == 'Critical' else '🟡'}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Action buttons for each alert
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"✅ Ack", key=f"ack_{alert['id']}", use_container_width=True):
                    st.success(f"Alert {alert['id']} acknowledged")
            with col2:
                if st.button(f"📞 Call", key=f"call_{alert['id']}", use_container_width=True):
                    st.info(f"Emergency services contacted for {alert['camera']}")
            with col3:
                if st.button(f"👁️ View", key=f"view_{alert['id']}", use_container_width=True):
                    st.info(f"Switching to {alert['camera']} live feed")
    
    def show_notification_center(self):
        """Real-time notification center"""
        st.markdown("### 📢 Notifications")
        
        # Notification settings
        with st.expander("🔧 Notification Settings"):
            email_alerts = st.checkbox("Email Alerts", value=True)
            sms_alerts = st.checkbox("SMS Alerts", value=False)
            push_notifications = st.checkbox("Push Notifications", value=True)
            sound_alerts = st.checkbox("Sound Alerts", value=True)
            
            alert_threshold = st.selectbox(
                "Minimum Alert Level",
                ["All", "Warning+", "Critical Only"],
                index=1
            )
        
        # Recent notifications
        st.markdown("**Recent Notifications**")
        
        notifications = [
            {
                'time': datetime.now() - timedelta(minutes=1),
                'type': 'Critical Alert',
                'message': 'High density detected at CAM_004',
                'status': 'unread'
            },
            {
                'time': datetime.now() - timedelta(minutes=3),
                'type': 'System Update',
                'message': 'AI model accuracy improved to 97.2%',
                'status': 'read'
            },
            {
                'time': datetime.now() - timedelta(minutes=7),
                'type': 'Warning Alert',
                'message': 'Moderate crowding at CAM_002',
                'status': 'read'
            }
        ]
        
        for notif in notifications:
            status_icon = "🔴" if notif['status'] == 'unread' else "✅"
            opacity = "1.0" if notif['status'] == 'unread' else "0.6"
            
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); padding: 0.8rem; 
                        border-radius: 6px; margin: 0.3rem 0; opacity: {opacity};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>{notif['type']}</strong> {status_icon}
                        <br><small>{notif['message']}</small>
                        <br><small style="color: rgba(255,255,255,0.6);">
                            {notif['time'].strftime('%H:%M:%S')}
                        </small>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick actions
        st.markdown("**Quick Actions**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔕 Mute All", use_container_width=True):
                st.info("All notifications muted for 30 minutes")
        with col2:
            if st.button("📤 Export Log", use_container_width=True):
                st.success("Notification log exported")

def show_real_time_alert_interface():
    """Main function to display the real-time alert interface"""
    alert_system = RealTimeAlertSystem()
    alert_system.show_live_alert_dashboard()