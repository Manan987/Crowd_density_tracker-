import streamlit as st
import json
from datetime import datetime

def show_settings():
    """Settings and configuration page"""
    st.title("⚙️ System Settings")
    
    # Get system components from session state
    if 'alert_system' not in st.session_state:
        st.error("Alert system not initialized")
        return
    
    alert_system = st.session_state.alert_system
    data_manager = st.session_state.data_manager
    
    # Settings tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚨 Alert Settings", 
        "📹 Camera Config", 
        "📧 Notifications", 
        "💾 Data Management", 
        "🔧 Advanced"
    ])
    
    with tab1:
        show_alert_settings(alert_system)
    
    with tab2:
        show_camera_settings()
    
    with tab3:
        show_notification_settings()
    
    with tab4:
        show_data_management_settings(data_manager)
    
    with tab5:
        show_advanced_settings()

def show_alert_settings(alert_system):
    """Alert system configuration"""
    st.header("Alert Threshold Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Threshold Values")
        
        # Current thresholds
        warning_threshold = st.number_input(
            "Warning Threshold (people/m²)",
            min_value=0.1,
            max_value=20.0,
            value=alert_system.warning_threshold,
            step=0.1,
            help="Density level that triggers warning alerts"
        )
        
        critical_threshold = st.number_input(
            "Critical Threshold (people/m²)",
            min_value=0.1,
            max_value=20.0,
            value=alert_system.critical_threshold,
            step=0.1,
            help="Density level that triggers critical alerts"
        )
        
        alert_duration = st.number_input(
            "Alert Duration (seconds)",
            min_value=1,
            max_value=300,
            value=alert_system.alert_duration,
            step=1,
            help="Minimum time between consecutive alerts"
        )
        
        # Validation
        if warning_threshold >= critical_threshold:
            st.error("Warning threshold must be less than critical threshold")
        else:
            if st.button("Update Alert Thresholds", type="primary"):
                try:
                    alert_system.update_thresholds(
                        warning_threshold, 
                        critical_threshold, 
                        alert_duration
                    )
                    st.success("Alert thresholds updated successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error updating thresholds: {e}")
    
    with col2:
        st.subheader("Alert Behavior")
        
        # Alert escalation settings
        enable_escalation = st.checkbox(
            "Enable Alert Escalation",
            value=True,
            help="Automatically escalate unacknowledged alerts"
        )
        
        if enable_escalation:
            escalation_time = st.number_input(
                "Escalation Time (minutes)",
                min_value=1,
                max_value=60,
                value=5,
                help="Time before unacknowledged alerts are escalated"
            )
        
        # Auto-acknowledgment
        auto_ack_normal = st.checkbox(
            "Auto-acknowledge when returning to normal",
            value=True,
            help="Automatically acknowledge alerts when density returns to normal"
        )
        
        # Alert suppression
        suppress_duplicates = st.checkbox(
            "Suppress duplicate alerts",
            value=True,
            help="Prevent repeated alerts for the same condition"
        )
    
    # Alert testing
    st.markdown("---")
    st.subheader("Alert Testing")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Test Warning Alert"):
            st.warning("🟡 WARNING: Test alert triggered!")
            st.info("This is how warning alerts will appear")
    
    with col2:
        if st.button("Test Critical Alert"):
            st.error("🔴 CRITICAL: Test alert triggered!")
            st.info("This is how critical alerts will appear")
    
    with col3:
        if st.button("Test Normal Status"):
            st.success("🟢 NORMAL: System operating normally")
            st.info("This is how normal status appears")

def show_camera_settings():
    """Camera configuration settings"""
    st.header("Camera Configuration")
    
    # Camera locations and settings
    cameras = ["CAM_001", "CAM_002", "CAM_003", "CAM_004"]
    
    for camera_id in cameras:
        with st.expander(f"📹 {camera_id} Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Basic settings
                location = st.text_input(
                    "Location",
                    value=f"Location {camera_id[-1]}",
                    key=f"location_{camera_id}"
                )
                
                enabled = st.checkbox(
                    "Enabled",
                    value=True,
                    key=f"enabled_{camera_id}"
                )
                
                # Monitoring zones
                monitor_zones = st.multiselect(
                    "Monitoring Zones",
                    ["Entrance", "Main Area", "Exit", "Stage", "Food Court"],
                    default=["Main Area"],
                    key=f"zones_{camera_id}"
                )
            
            with col2:
                # Technical settings
                resolution = st.selectbox(
                    "Resolution",
                    ["1920x1080", "1280x720", "640x480"],
                    index=1,
                    key=f"resolution_{camera_id}"
                )
                
                fps = st.selectbox(
                    "Frame Rate (FPS)",
                    [30, 25, 15, 10],
                    index=2,
                    key=f"fps_{camera_id}"
                )
                
                # Custom density calculation area
                area_m2 = st.number_input(
                    "Monitoring Area (m²)",
                    min_value=1.0,
                    max_value=10000.0,
                    value=100.0,
                    step=10.0,
                    key=f"area_{camera_id}",
                    help="Physical area covered by this camera"
                )
    
    # Global camera settings
    st.markdown("---")
    st.subheader("Global Camera Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Processing settings
        st.write("**Processing Options**")
        
        enable_motion_detection = st.checkbox(
            "Enable Motion Detection",
            value=True,
            help="Use motion detection to improve accuracy"
        )
        
        enable_night_mode = st.checkbox(
            "Enable Night Mode",
            value=False,
            help="Optimize processing for low-light conditions"
        )
        
        processing_quality = st.selectbox(
            "Processing Quality",
            ["High", "Medium", "Low", "Auto"],
            index=1,
            help="Higher quality uses more computational resources"
        )
    
    with col2:
        # Calibration settings
        st.write("**Calibration Options**")
        
        auto_calibration = st.checkbox(
            "Auto Calibration",
            value=True,
            help="Automatically calibrate cameras based on known reference points"
        )
        
        if not auto_calibration:
            manual_scale = st.number_input(
                "Manual Scale (pixels/meter)",
                min_value=1.0,
                max_value=1000.0,
                value=100.0,
                help="Pixels per meter for manual calibration"
            )
        
        calibration_interval = st.selectbox(
            "Calibration Interval",
            ["Hourly", "Daily", "Weekly", "Manual"],
            index=1,
            help="How often to recalibrate cameras"
        )
    
    # Save settings
    if st.button("Save Camera Settings", type="primary"):
        st.success("Camera settings saved successfully!")

def show_notification_settings():
    """Notification system configuration"""
    st.header("Notification Settings")
    
    # Notification channels
    st.subheader("Notification Channels")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Email notifications
        st.write("**📧 Email Notifications**")
        
        email_enabled = st.checkbox("Enable Email Notifications", value=True)
        
        if email_enabled:
            email_addresses = st.text_area(
                "Recipient Email Addresses",
                value="security@venue.com\nmanager@venue.com\noperator@venue.com",
                help="Enter one email address per line"
            )
            
            email_priority = st.selectbox(
                "Email Alert Levels",
                ["All Alerts", "Warning and Critical", "Critical Only"],
                index=1
            )
        
        # SMS notifications
        st.write("**📱 SMS Notifications**")
        
        sms_enabled = st.checkbox("Enable SMS Notifications", value=False)
        
        if sms_enabled:
            phone_numbers = st.text_area(
                "Phone Numbers",
                value="+1234567890\n+0987654321",
                help="Enter phone numbers in international format, one per line"
            )
            
            sms_priority = st.selectbox(
                "SMS Alert Levels",
                ["All Alerts", "Warning and Critical", "Critical Only"],
                index=2,
                key="sms_priority"
            )
    
    with col2:
        # Push notifications and integrations
        st.write("**🔔 Push Notifications**")
        
        push_enabled = st.checkbox("Enable Push Notifications", value=True)
        
        if push_enabled:
            notification_sound = st.checkbox("Enable Notification Sound", value=True)
            browser_notifications = st.checkbox("Enable Browser Notifications", value=True)
        
        # Third-party integrations
        st.write("**🔗 Integrations**")
        
        slack_enabled = st.checkbox("Slack Integration", value=False)
        if slack_enabled:
            slack_webhook = st.text_input(
                "Slack Webhook URL",
                placeholder="https://hooks.slack.com/services/...",
                type="password"
            )
        
        teams_enabled = st.checkbox("Microsoft Teams Integration", value=False)
        if teams_enabled:
            teams_webhook = st.text_input(
                "Teams Webhook URL",
                placeholder="https://outlook.office.com/webhook/...",
                type="password"
            )
    
    # Notification timing and frequency
    st.markdown("---")
    st.subheader("Notification Timing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rate limiting
        max_notifications_per_hour = st.number_input(
            "Max Notifications per Hour",
            min_value=1,
            max_value=100,
            value=10,
            help="Limit to prevent notification spam"
        )
        
        notification_delay = st.number_input(
            "Delay Between Notifications (seconds)",
            min_value=0,
            max_value=300,
            value=30,
            help="Minimum time between notifications for the same alert"
        )
    
    with col2:
        # Quiet hours
        enable_quiet_hours = st.checkbox("Enable Quiet Hours", value=False)
        
        if enable_quiet_hours:
            quiet_start = st.time_input("Quiet Hours Start", value=datetime.strptime("22:00", "%H:%M").time())
            quiet_end = st.time_input("Quiet Hours End", value=datetime.strptime("08:00", "%H:%M").time())
            
            quiet_critical_override = st.checkbox(
                "Send Critical Alerts During Quiet Hours",
                value=True,
                help="Override quiet hours for critical alerts"
            )
    
    # Test notifications
    st.markdown("---")
    st.subheader("Test Notifications")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Test Email"):
            st.info("📧 Test email sent!")
    
    with col2:
        if st.button("Test SMS"):
            st.info("📱 Test SMS sent!")
    
    with col3:
        if st.button("Test Push Notification"):
            st.info("🔔 Test push notification sent!")
    
    # Save notification settings
    if st.button("Save Notification Settings", type="primary"):
        st.success("Notification settings saved successfully!")

def show_data_management_settings(data_manager):
    """Data management and retention settings"""
    st.header("Data Management")
    
    # Data retention settings
    st.subheader("Data Retention")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Retention periods
        density_retention = st.number_input(
            "Density Data Retention (days)",
            min_value=1,
            max_value=365,
            value=30,
            help="How long to keep density measurement data"
        )
        
        alert_retention = st.number_input(
            "Alert Data Retention (days)",
            min_value=1,
            max_value=365,
            value=90,
            help="How long to keep alert event data"
        )
        
        log_retention = st.number_input(
            "System Log Retention (days)",
            min_value=1,
            max_value=365,
            value=14,
            help="How long to keep system logs"
        )
    
    with col2:
        # Storage management
        st.write("**Storage Management**")
        
        auto_cleanup = st.checkbox(
            "Enable Automatic Cleanup",
            value=True,
            help="Automatically delete old data based on retention settings"
        )
        
        if auto_cleanup:
            cleanup_schedule = st.selectbox(
                "Cleanup Schedule",
                ["Daily", "Weekly", "Monthly"],
                index=1
            )
        
        compress_old_data = st.checkbox(
            "Compress Old Data",
            value=True,
            help="Compress data older than 7 days to save space"
        )
    
    # Database information
    st.markdown("---")
    st.subheader("Database Information")
    
    db_info = data_manager.get_database_info()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Database Size", f"{db_info.get('database_size_bytes', 0) / 1024:.1f} KB")
    
    with col2:
        density_records = db_info.get('table_counts', {}).get('density_data', 0)
        st.metric("Density Records", density_records)
    
    with col3:
        alert_records = db_info.get('table_counts', {}).get('alert_events', 0)
        st.metric("Alert Records", alert_records)
    
    with col4:
        log_records = db_info.get('table_counts', {}).get('system_logs', 0)
        st.metric("Log Records", log_records)
    
    # Data operations
    st.markdown("---")
    st.subheader("Data Operations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export data
        st.write("**Export Data**")
        
        export_format = st.selectbox("Export Format", ["CSV", "JSON"], key="export_format")
        export_period = st.selectbox(
            "Export Period",
            ["Last 24 Hours", "Last Week", "Last Month", "All Data"],
            key="export_period"
        )
        
        if st.button("Export Data"):
            hours_map = {
                "Last 24 Hours": 24,
                "Last Week": 168,
                "Last Month": 720,
                "All Data": 8760  # 1 year
            }
            
            hours = hours_map[export_period]
            export_data = data_manager.export_data(hours=hours, format=export_format.lower())
            
            filename = f"crowd_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format.lower()}"
            
            st.download_button(
                f"Download {export_format}",
                export_data,
                file_name=filename,
                mime=f"text/{export_format.lower()}"
            )
    
    with col2:
        # Backup database
        st.write("**Database Backup**")
        
        if st.button("Create Backup"):
            backup_filename = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            success = data_manager.backup_database(backup_filename)
            
            if success:
                st.success("Backup created successfully!")
            else:
                st.error("Failed to create backup")
    
    with col3:
        # Clear data
        st.write("**Clear Data**")
        
        clear_option = st.selectbox(
            "Clear Option",
            ["Last 7 Days", "Last 30 Days", "All Data"],
            key="clear_option"
        )
        
        if st.button("Clear Data", type="secondary"):
            days_map = {
                "Last 7 Days": 7,
                "Last 30 Days": 30,
                "All Data": None
            }
            
            days = days_map[clear_option]
            
            if days:
                data_manager.clear_data(older_than_days=days)
                st.success(f"Cleared data older than {days} days")
            else:
                if st.session_state.get('confirm_clear_all'):
                    data_manager.clear_data()
                    st.success("All data cleared successfully!")
                    st.session_state.confirm_clear_all = False
                    st.rerun()
                else:
                    st.session_state.confirm_clear_all = True
                    st.warning("Click again to confirm clearing ALL data")

def show_advanced_settings():
    """Advanced system settings"""
    st.header("Advanced Settings")
    
    # Performance settings
    st.subheader("Performance Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ML Model settings
        st.write("**ML Model Configuration**")
        
        model_precision = st.selectbox(
            "Model Precision",
            ["Float32", "Float16", "Int8"],
            index=0,
            help="Lower precision uses less memory but may reduce accuracy"
        )
        
        batch_size = st.number_input(
            "Processing Batch Size",
            min_value=1,
            max_value=32,
            value=4,
            help="Number of frames processed simultaneously"
        )
        
        enable_gpu = st.checkbox(
            "Enable GPU Acceleration",
            value=False,
            help="Use GPU for model inference (requires compatible hardware)"
        )
    
    with col2:
        # System performance
        st.write("**System Performance**")
        
        processing_threads = st.number_input(
            "Processing Threads",
            min_value=1,
            max_value=16,
            value=4,
            help="Number of parallel processing threads"
        )
        
        memory_limit_mb = st.number_input(
            "Memory Limit (MB)",
            min_value=512,
            max_value=8192,
            value=2048,
            help="Maximum memory usage for processing"
        )
        
        enable_caching = st.checkbox(
            "Enable Result Caching",
            value=True,
            help="Cache processing results to improve performance"
        )
    
    # Security settings
    st.markdown("---")
    st.subheader("Security Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Access control
        st.write("**Access Control**")
        
        enable_auth = st.checkbox(
            "Enable Authentication",
            value=False,
            help="Require login to access the system"
        )
        
        if enable_auth:
            session_timeout = st.number_input(
                "Session Timeout (minutes)",
                min_value=5,
                max_value=480,
                value=60
            )
        
        enable_audit_log = st.checkbox(
            "Enable Audit Logging",
            value=True,
            help="Log all user actions for security auditing"
        )
    
    with col2:
        # Privacy settings
        st.write("**Privacy Settings**")
        
        anonymize_data = st.checkbox(
            "Anonymize Stored Data",
            value=True,
            help="Remove identifying information from stored data"
        )
        
        data_encryption = st.checkbox(
            "Enable Data Encryption",
            value=False,
            help="Encrypt sensitive data at rest"
        )
        
        if data_encryption:
            encryption_key = st.text_input(
                "Encryption Key",
                type="password",
                help="Key used for data encryption"
            )
    
    # System maintenance
    st.markdown("---")
    st.subheader("System Maintenance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Maintenance schedule
        auto_maintenance = st.checkbox(
            "Enable Automatic Maintenance",
            value=True,
            help="Automatically perform system maintenance tasks"
        )
        
        if auto_maintenance:
            maintenance_time = st.time_input(
                "Maintenance Time",
                value=datetime.strptime("02:00", "%H:%M").time(),
                help="Time of day to perform maintenance"
            )
    
    with col2:
        # Update settings
        auto_updates = st.checkbox(
            "Enable Automatic Updates",
            value=False,
            help="Automatically install system updates"
        )
        
        if auto_updates:
            update_channel = st.selectbox(
                "Update Channel",
                ["Stable", "Beta", "Development"],
                index=0
            )
    
    with col3:
        # Monitoring
        enable_monitoring = st.checkbox(
            "Enable System Monitoring",
            value=True,
            help="Monitor system health and performance"
        )
        
        if enable_monitoring:
            monitoring_interval = st.number_input(
                "Monitoring Interval (seconds)",
                min_value=10,
                max_value=300,
                value=60
            )
    
    # Debug settings
    st.markdown("---")
    st.subheader("Debug Settings")
    
    debug_mode = st.checkbox(
        "Enable Debug Mode",
        value=False,
        help="Enable detailed logging and debug information"
    )
    
    if debug_mode:
        col1, col2 = st.columns(2)
        
        with col1:
            log_level = st.selectbox(
                "Log Level",
                ["DEBUG", "INFO", "WARNING", "ERROR"],
                index=1
            )
        
        with col2:
            save_debug_frames = st.checkbox(
                "Save Debug Frames",
                value=False,
                help="Save processed video frames for debugging"
            )
    
    # Save advanced settings
    if st.button("Save Advanced Settings", type="primary"):
        st.success("Advanced settings saved successfully!")
        st.info("Some settings may require system restart to take effect")
