import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional
from datetime import datetime, timedelta
import json
import sqlite3
from dataclasses import dataclass
from enum import Enum
# import smtplib
# from email.mime.text import MimeText
# from email.mime.multipart import MimeMultipart

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class AlertType(Enum):
    DENSITY_THRESHOLD = "DENSITY_THRESHOLD"
    RAPID_INCREASE = "RAPID_INCREASE"
    SUSTAINED_HIGH = "SUSTAINED_HIGH"
    PATTERN_ANOMALY = "PATTERN_ANOMALY"

@dataclass
class Alert:
    alert_id: str
    stream_id: str
    alert_type: AlertType
    risk_level: RiskLevel
    timestamp: datetime
    crowd_count: int
    max_density: float
    confidence: float
    message: str
    metadata: Dict

class EnhancedAlertSystem:
    """Advanced alert system with ML-based risk assessment and multiple notification channels"""
    
    def __init__(self, db_path: str = "alerts.db"):
        self.db_path = db_path
        self.alert_history = []
        self.notification_callbacks = []
        
        # Thresholds (configurable)
        self.thresholds = {
            'density_high': 0.7,
            'density_critical': 0.9,
            'count_high': 100,
            'count_critical': 200,
            'rapid_increase_rate': 0.5,  # 50% increase in 30 seconds
            'sustained_duration': 60,    # 60 seconds
        }
        
        # Historical data for pattern analysis
        self.density_history = {}
        self.pattern_analyzer = PatternAnalyzer()
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for alert storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                stream_id TEXT,
                alert_type TEXT,
                risk_level TEXT,
                timestamp TEXT,
                crowd_count INTEGER,
                max_density REAL,
                confidence REAL,
                message TEXT,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS density_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stream_id TEXT,
                timestamp TEXT,
                crowd_count INTEGER,
                max_density REAL,
                avg_density REAL,
                confidence REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def process_density_result(self, stream_id: str, result: Dict) -> List[Alert]:
        """Process density estimation result and generate alerts if needed"""
        alerts = []
        
        # Store in history
        self._store_density_data(stream_id, result)
        
        # Update historical data
        if stream_id not in self.density_history:
            self.density_history[stream_id] = []
        
        self.density_history[stream_id].append({
            'timestamp': datetime.now(),
            'count': result['total_count'],
            'max_density': result['max_density'],
            'avg_density': result['avg_density'],
            'confidence': result['confidence_score']
        })
        
        # Keep only recent history (last 10 minutes)
        cutoff_time = datetime.now() - timedelta(minutes=10)
        self.density_history[stream_id] = [
            entry for entry in self.density_history[stream_id]
            if entry['timestamp'] > cutoff_time
        ]
        
        # Check for various alert conditions
        alerts.extend(self._check_density_thresholds(stream_id, result))
        alerts.extend(self._check_rapid_increase(stream_id))
        alerts.extend(self._check_sustained_high_density(stream_id))
        alerts.extend(self._check_pattern_anomalies(stream_id))
        
        # Process and store alerts
        for alert in alerts:
            self._store_alert(alert)
            self._send_notifications(alert)
        
        return alerts
    
    def _check_density_thresholds(self, stream_id: str, result: Dict) -> List[Alert]:
        """Check for density threshold violations"""
        alerts = []
        
        count = result['total_count']
        max_density = result['max_density']
        confidence = result['confidence_score']
        
        # Critical density alert
        if (max_density > self.thresholds['density_critical'] or 
            count > self.thresholds['count_critical']) and confidence > 0.7:
            
            alert = Alert(
                alert_id=self._generate_alert_id(),
                stream_id=stream_id,
                alert_type=AlertType.DENSITY_THRESHOLD,
                risk_level=RiskLevel.CRITICAL,
                timestamp=datetime.now(),
                crowd_count=count,
                max_density=max_density,
                confidence=confidence,
                message=f"CRITICAL: Extremely high crowd density detected (Count: {count}, Max Density: {max_density:.2f})",
                metadata=result
            )
            alerts.append(alert)
        
        # High density alert
        elif (max_density > self.thresholds['density_high'] or 
              count > self.thresholds['count_high']) and confidence > 0.6:
            
            alert = Alert(
                alert_id=self._generate_alert_id(),
                stream_id=stream_id,
                alert_type=AlertType.DENSITY_THRESHOLD,
                risk_level=RiskLevel.HIGH,
                timestamp=datetime.now(),
                crowd_count=count,
                max_density=max_density,
                confidence=confidence,
                message=f"HIGH: High crowd density detected (Count: {count}, Max Density: {max_density:.2f})",
                metadata=result
            )
            alerts.append(alert)
        
        return alerts
    
    def _check_rapid_increase(self, stream_id: str) -> List[Alert]:
        """Check for rapid increase in crowd density"""
        alerts = []
        
        if stream_id not in self.density_history or len(self.density_history[stream_id]) < 2:
            return alerts
        
        history = self.density_history[stream_id]
        
        # Check last 30 seconds
        cutoff_time = datetime.now() - timedelta(seconds=30)
        recent_data = [entry for entry in history if entry['timestamp'] > cutoff_time]
        
        if len(recent_data) >= 2:
            initial_count = recent_data[0]['count']
            current_count = recent_data[-1]['count']
            
            if initial_count > 0:
                increase_rate = (current_count - initial_count) / initial_count
                
                if increase_rate > self.thresholds['rapid_increase_rate']:
                    alert = Alert(
                        alert_id=self._generate_alert_id(),
                        stream_id=stream_id,
                        alert_type=AlertType.RAPID_INCREASE,
                        risk_level=RiskLevel.HIGH,
                        timestamp=datetime.now(),
                        crowd_count=current_count,
                        max_density=recent_data[-1]['max_density'],
                        confidence=recent_data[-1]['confidence'],
                        message=f"RAPID INCREASE: Crowd size increased by {increase_rate*100:.1f}% in 30 seconds",
                        metadata={'increase_rate': increase_rate, 'initial_count': initial_count}
                    )
                    alerts.append(alert)
        
        return alerts
    
    def _check_sustained_high_density(self, stream_id: str) -> List[Alert]:
        """Check for sustained high density over time"""
        alerts = []
        
        if stream_id not in self.density_history:
            return alerts
        
        history = self.density_history[stream_id]
        
        # Check last minute
        cutoff_time = datetime.now() - timedelta(seconds=self.thresholds['sustained_duration'])
        recent_data = [entry for entry in history if entry['timestamp'] > cutoff_time]
        
        if len(recent_data) >= 5:  # At least 5 data points
            high_density_count = sum(1 for entry in recent_data 
                                   if entry['max_density'] > self.thresholds['density_high'])
            
            if high_density_count / len(recent_data) > 0.8:  # 80% of time
                avg_count = np.mean([entry['count'] for entry in recent_data])
                avg_density = np.mean([entry['max_density'] for entry in recent_data])
                
                alert = Alert(
                    alert_id=self._generate_alert_id(),
                    stream_id=stream_id,
                    alert_type=AlertType.SUSTAINED_HIGH,
                    risk_level=RiskLevel.HIGH,
                    timestamp=datetime.now(),
                    crowd_count=int(avg_count),
                    max_density=avg_density,
                    confidence=np.mean([entry['confidence'] for entry in recent_data]),
                    message=f"SUSTAINED HIGH: High density maintained for {self.thresholds['sustained_duration']} seconds",
                    metadata={'duration': self.thresholds['sustained_duration'], 'avg_density': avg_density}
                )
                alerts.append(alert)
        
        return alerts
    
    def _check_pattern_anomalies(self, stream_id: str) -> List[Alert]:
        """Check for unusual patterns using ML-based analysis"""
        alerts = []
        
        if stream_id not in self.density_history or len(self.density_history[stream_id]) < 10:
            return alerts
        
        # Use pattern analyzer to detect anomalies
        anomaly_score = self.pattern_analyzer.detect_anomaly(self.density_history[stream_id])
        
        if anomaly_score > 0.8:  # High anomaly score
            latest_data = self.density_history[stream_id][-1]
            
            alert = Alert(
                alert_id=self._generate_alert_id(),
                stream_id=stream_id,
                alert_type=AlertType.PATTERN_ANOMALY,
                risk_level=RiskLevel.MEDIUM,
                timestamp=datetime.now(),
                crowd_count=latest_data['count'],
                max_density=latest_data['max_density'],
                confidence=latest_data['confidence'],
                message=f"PATTERN ANOMALY: Unusual crowd behavior detected (Anomaly Score: {anomaly_score:.2f})",
                metadata={'anomaly_score': anomaly_score}
            )
            alerts.append(alert)
        
        return alerts
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        import uuid
        return str(uuid.uuid4())
    
    def _store_alert(self, alert: Alert):
        """Store alert in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert.alert_id,
            alert.stream_id,
            alert.alert_type.value,
            alert.risk_level.value,
            alert.timestamp.isoformat(),
            alert.crowd_count,
            alert.max_density,
            alert.confidence,
            alert.message,
            json.dumps(alert.metadata)
        ))
        
        conn.commit()
        conn.close()
        
        # Add to in-memory history
        self.alert_history.append(alert)
    
    def _store_density_data(self, stream_id: str, result: Dict):
        """Store density data in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO density_history (stream_id, timestamp, crowd_count, max_density, avg_density, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            stream_id,
            datetime.now().isoformat(),
            result['total_count'],
            result['max_density'],
            result['avg_density'],
            result['confidence_score']
        ))
        
        conn.commit()
        conn.close()
    
    def _send_notifications(self, alert: Alert):
        """Send notifications through registered callbacks"""
        for callback in self.notification_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Error in notification callback: {e}")
    
    def add_notification_callback(self, callback: Callable[[Alert], None]):
        """Add notification callback"""
        self.notification_callbacks.append(callback)
    
    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """Get recent alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp > cutoff_time]
    
    def update_thresholds(self, new_thresholds: Dict):
        """Update alert thresholds"""
        self.thresholds.update(new_thresholds)

class PatternAnalyzer:
    """Simple pattern analyzer for detecting anomalies"""
    
    def detect_anomaly(self, history: List[Dict]) -> float:
        """Detect anomalies in crowd density patterns"""
        if len(history) < 10:
            return 0.0
        
        # Extract features
        counts = [entry['count'] for entry in history]
        densities = [entry['max_density'] for entry in history]
        
        # Calculate statistics
        count_mean = np.mean(counts)
        count_std = np.std(counts)
        density_mean = np.mean(densities)
        density_std = np.std(densities)
        
        # Check latest values against historical patterns
        latest_count = counts[-1]
        latest_density = densities[-1]
        
        # Calculate z-scores
        count_zscore = abs(latest_count - count_mean) / (count_std + 1e-6)
        density_zscore = abs(latest_density - density_mean) / (density_std + 1e-6)
        
        # Combine scores
        anomaly_score = min((count_zscore + density_zscore) / 4.0, 1.0)
        
        return anomaly_score