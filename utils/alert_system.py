import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
import queue

class AlertSystem:
    """
    Alert system for crowd density monitoring
    Manages thresholds, alert levels, and notifications
    """
    
    def __init__(self):
        # Default thresholds (people per square meter)
        self.warning_threshold = 2.0
        self.critical_threshold = 4.0
        self.alert_duration = 5  # seconds
        
        # Alert state management
        self.current_alert_level = "Normal"
        self.last_alert_time = None
        self.alert_history = []
        self.active_alerts = {}
        
        # Notification queue
        self.notification_queue = queue.Queue()
        self.notification_thread = None
        self.is_notifying = False
        
        # Alert statistics
        self.alert_counts = {
            "Normal": 0,
            "Warning": 0,
            "Critical": 0
        }
        
    def check_alert_level(self, density: float, camera_id: str = "default") -> str:
        """
        Check current alert level based on density
        
        Args:
            density: Current crowd density (people/m²)
            camera_id: Camera identifier
            
        Returns:
            Alert level string
        """
        current_time = datetime.now()
        
        # Determine alert level
        if density >= self.critical_threshold:
            alert_level = "Critical"
        elif density >= self.warning_threshold:
            alert_level = "Warning"
        else:
            alert_level = "Normal"
        
        # Check if alert level has changed
        previous_level = self.active_alerts.get(camera_id, "Normal")
        
        if alert_level != previous_level:
            self._handle_alert_change(camera_id, previous_level, alert_level, density)
        
        # Update active alerts
        self.active_alerts[camera_id] = alert_level
        self.current_alert_level = self._get_highest_alert_level()
        
        # Update statistics
        self.alert_counts[alert_level] += 1
        
        return alert_level
    
    def _handle_alert_change(self, camera_id: str, previous_level: str, 
                           new_level: str, density: float):
        """Handle alert level changes"""
        timestamp = datetime.now()
        
        # Log alert change
        alert_record = {
            "timestamp": timestamp,
            "camera_id": camera_id,
            "previous_level": previous_level,
            "new_level": new_level,
            "density": density,
            "acknowledged": False
        }
        
        self.alert_history.append(alert_record)
        
        # Trigger notifications for escalating alerts
        if self._is_escalating_alert(previous_level, new_level):
            self._queue_notification(alert_record)
        
        # Update last alert time
        if new_level != "Normal":
            self.last_alert_time = timestamp
    
    def _is_escalating_alert(self, previous_level: str, new_level: str) -> bool:
        """Check if alert is escalating"""
        alert_hierarchy = {"Normal": 0, "Warning": 1, "Critical": 2}
        return alert_hierarchy.get(new_level, 0) > alert_hierarchy.get(previous_level, 0)
    
    def _get_highest_alert_level(self) -> str:
        """Get the highest alert level across all cameras"""
        if not self.active_alerts:
            return "Normal"
        
        alert_hierarchy = {"Normal": 0, "Warning": 1, "Critical": 2}
        highest_level = max(self.active_alerts.values(), 
                          key=lambda x: alert_hierarchy.get(x, 0))
        return highest_level
    
    def _queue_notification(self, alert_record: dict):
        """Queue notification for sending"""
        try:
            self.notification_queue.put(alert_record, timeout=1)
            
            # Start notification thread if not running
            if not self.is_notifying:
                self._start_notification_thread()
                
        except queue.Full:
            print("Notification queue is full, dropping alert")
    
    def _start_notification_thread(self):
        """Start notification processing thread"""
        if self.notification_thread is None or not self.notification_thread.is_alive():
            self.notification_thread = threading.Thread(
                target=self._process_notifications,
                daemon=True
            )
            self.is_notifying = True
            self.notification_thread.start()
    
    def _process_notifications(self):
        """Process queued notifications"""
        while self.is_notifying:
            try:
                alert_record = self.notification_queue.get(timeout=1)
                self._send_notification(alert_record)
                self.notification_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing notification: {e}")
    
    def _send_notification(self, alert_record: dict):
        """Send notification (placeholder for actual implementation)"""
        # In a real implementation, this would send emails, SMS, etc.
        print(f"ALERT: {alert_record['new_level']} level detected on {alert_record['camera_id']}")
        print(f"Density: {alert_record['density']:.2f} people/m² at {alert_record['timestamp']}")
    
    def acknowledge_alert(self, camera_id: str, operator_id: str = "system") -> bool:
        """
        Acknowledge an active alert
        
        Args:
            camera_id: Camera with the alert
            operator_id: ID of the operator acknowledging
            
        Returns:
            Success status
        """
        try:
            # Find the most recent unacknowledged alert for this camera
            for alert in reversed(self.alert_history):
                if (alert["camera_id"] == camera_id and 
                    not alert["acknowledged"] and 
                    alert["new_level"] != "Normal"):
                    
                    alert["acknowledged"] = True
                    alert["acknowledged_by"] = operator_id
                    alert["acknowledged_at"] = datetime.now()
                    
                    print(f"Alert acknowledged for {camera_id} by {operator_id}")
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error acknowledging alert: {e}")
            return False
    
    def update_thresholds(self, warning: float, critical: float, duration: int):
        """
        Update alert thresholds
        
        Args:
            warning: Warning threshold
            critical: Critical threshold  
            duration: Alert duration in seconds
        """
        if warning >= critical:
            raise ValueError("Warning threshold must be less than critical threshold")
        
        if warning <= 0 or critical <= 0:
            raise ValueError("Thresholds must be positive values")
        
        self.warning_threshold = warning
        self.critical_threshold = critical
        self.alert_duration = duration
        
        print(f"Thresholds updated: Warning={warning}, Critical={critical}, Duration={duration}s")
    
    def get_alert_summary(self, hours: int = 24) -> Dict:
        """
        Get alert summary for specified time period
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Alert summary dictionary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [
            alert for alert in self.alert_history 
            if alert["timestamp"] >= cutoff_time
        ]
        
        # Count alerts by level
        level_counts = {"Warning": 0, "Critical": 0}
        for alert in recent_alerts:
            if alert["new_level"] in level_counts:
                level_counts[alert["new_level"]] += 1
        
        # Count alerts by camera
        camera_counts = {}
        for alert in recent_alerts:
            camera_id = alert["camera_id"]
            camera_counts[camera_id] = camera_counts.get(camera_id, 0) + 1
        
        # Calculate average response time (for acknowledged alerts)
        acknowledged_alerts = [
            alert for alert in recent_alerts 
            if alert.get("acknowledged", False)
        ]
        
        avg_response_time = 0
        if acknowledged_alerts:
            response_times = []
            for alert in acknowledged_alerts:
                if "acknowledged_at" in alert:
                    response_time = (alert["acknowledged_at"] - alert["timestamp"]).total_seconds()
                    response_times.append(response_time)
            
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
        
        return {
            "time_period_hours": hours,
            "total_alerts": len(recent_alerts),
            "alerts_by_level": level_counts,
            "alerts_by_camera": camera_counts,
            "acknowledged_count": len(acknowledged_alerts),
            "average_response_time_seconds": avg_response_time,
            "current_alert_level": self.current_alert_level,
            "active_cameras_with_alerts": len([
                camera for camera, level in self.active_alerts.items() 
                if level != "Normal"
            ])
        }
    
    def get_recent_alerts(self, count: int = 10) -> List[Dict]:
        """Get recent alert records"""
        return self.alert_history[-count:] if self.alert_history else []
    
    def clear_old_alerts(self, days: int = 30):
        """Clear alert history older than specified days"""
        cutoff_time = datetime.now() - timedelta(days=days)
        self.alert_history = [
            alert for alert in self.alert_history 
            if alert["timestamp"] >= cutoff_time
        ]
        print(f"Cleared alerts older than {days} days")
    
    def export_alert_data(self) -> str:
        """Export alert data as CSV string"""
        import csv
        import io
        
        output = io.StringIO()
        
        if not self.alert_history:
            return "No alert data available"
        
        fieldnames = [
            "timestamp", "camera_id", "previous_level", "new_level", 
            "density", "acknowledged", "acknowledged_by", "acknowledged_at"
        ]
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for alert in self.alert_history:
            writer.writerow(alert)
        
        return output.getvalue()
    
    def stop_notifications(self):
        """Stop notification processing"""
        self.is_notifying = False
        if self.notification_thread and self.notification_thread.is_alive():
            self.notification_thread.join(timeout=2)
