import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from sqlalchemy import text, desc, func
from sqlalchemy.orm import Session
from database.models import (
    DatabaseManager, Camera, DensityReading, AlertEvent, 
    NotificationLog, SystemLog, ModelPerformance, Configuration
)
import json
import logging

class PostgreSQLDataService:
    """Enhanced data service using PostgreSQL database"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.logger = logging.getLogger(__name__)
        
        # Initialize database and default data
        try:
            self.db_manager.create_tables()
            self.db_manager.initialize_default_data()
        except Exception as e:
            self.logger.error(f"Database initialization error: {e}")
    
    def add_density_reading(self, camera_id: str, density_value: float, 
                           crowd_count: int, confidence_score: float = 0.95,
                           processing_time_ms: float = None) -> bool:
        """Add a new density reading to the database"""
        session = self.db_manager.get_session()
        try:
            reading = DensityReading(
                camera_id=camera_id,
                density_value=density_value,
                crowd_count=crowd_count,
                confidence_score=confidence_score,
                processing_time_ms=processing_time_ms,
                frame_quality_score=0.9,  # Default quality score
                timestamp=datetime.now()
            )
            
            session.add(reading)
            session.commit()
            
            # Log the activity
            self.add_system_log("INFO", "DATA_SERVICE", 
                              f"Added density reading: {density_value:.2f} people/m² for {camera_id}")
            
            return True
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error adding density reading: {e}")
            return False
        finally:
            session.close()
    
    def get_recent_density_data(self, minutes: int = 60, hours: int = None,
                               cameras: Optional[List[str]] = None) -> pd.DataFrame:
        """Get recent density readings"""
        session = self.db_manager.get_session()
        try:
            # Calculate time cutoff
            if hours is not None:
                cutoff_time = datetime.now() - timedelta(hours=hours)
            else:
                cutoff_time = datetime.now() - timedelta(minutes=minutes)
            
            # Build query
            query = session.query(DensityReading).filter(
                DensityReading.timestamp >= cutoff_time
            )
            
            # Filter by cameras if specified
            if cameras:
                query = query.filter(DensityReading.camera_id.in_(cameras))
            
            # Order by timestamp
            query = query.order_by(DensityReading.timestamp)
            
            # Execute query and convert to DataFrame
            readings = query.all()
            
            if not readings:
                return pd.DataFrame()
            
            data = []
            for reading in readings:
                data.append({
                    'timestamp': reading.timestamp,
                    'camera_id': reading.camera_id,
                    'density': reading.density_value,
                    'crowd_count': reading.crowd_count,
                    'confidence_score': reading.confidence_score,
                    'processing_time_ms': reading.processing_time_ms
                })
            
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting recent density data: {e}")
            return pd.DataFrame()
        finally:
            session.close()
    
    def add_alert_event(self, camera_id: str, alert_type: str, density_value: float,
                       threshold_exceeded: float = None) -> int:
        """Add a new alert event"""
        session = self.db_manager.get_session()
        try:
            alert = AlertEvent(
                camera_id=camera_id,
                alert_type=alert_type,
                density_value=density_value,
                threshold_exceeded=threshold_exceeded,
                created_at=datetime.now()
            )
            
            session.add(alert)
            session.commit()
            
            # Log the alert
            self.add_system_log("WARNING" if alert_type == "Warning" else "CRITICAL", 
                              "ALERT_SYSTEM", 
                              f"{alert_type} alert triggered for {camera_id}: {density_value:.2f} people/m²")
            
            return alert.id
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error adding alert event: {e}")
            return None
        finally:
            session.close()
    
    def get_alert_events(self, hours: int = 24, alert_types: List[str] = None) -> pd.DataFrame:
        """Get alert events from specified time period"""
        session = self.db_manager.get_session()
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            query = session.query(AlertEvent).filter(
                AlertEvent.created_at >= cutoff_time
            )
            
            if alert_types:
                query = query.filter(AlertEvent.alert_type.in_(alert_types))
            
            query = query.order_by(desc(AlertEvent.created_at))
            
            alerts = query.all()
            
            if not alerts:
                return pd.DataFrame()
            
            data = []
            for alert in alerts:
                data.append({
                    'id': alert.id,
                    'timestamp': alert.created_at,
                    'camera_id': alert.camera_id,
                    'alert_level': alert.alert_type,
                    'density': alert.density_value,
                    'threshold_exceeded': alert.threshold_exceeded,
                    'acknowledged': alert.acknowledged,
                    'acknowledged_by': alert.acknowledged_by,
                    'acknowledged_at': alert.acknowledged_at
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            self.logger.error(f"Error getting alert events: {e}")
            return pd.DataFrame()
        finally:
            session.close()
    
    def acknowledge_alert(self, alert_id: int, acknowledged_by: str) -> bool:
        """Acknowledge an alert event"""
        session = self.db_manager.get_session()
        try:
            alert = session.query(AlertEvent).filter(AlertEvent.id == alert_id).first()
            
            if alert:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                session.commit()
                
                self.add_system_log("INFO", "ALERT_SYSTEM", 
                                  f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True
            
            return False
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error acknowledging alert: {e}")
            return False
        finally:
            session.close()
    
    def get_density_statistics(self, hours: int = 24, camera_id: str = None) -> Dict:
        """Get comprehensive density statistics"""
        session = self.db_manager.get_session()
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            query = session.query(DensityReading).filter(
                DensityReading.timestamp >= cutoff_time
            )
            
            if camera_id:
                query = query.filter(DensityReading.camera_id == camera_id)
            
            readings = query.all()
            
            if not readings:
                return {}
            
            densities = [r.density_value for r in readings]
            counts = [r.crowd_count for r in readings]
            
            stats = {
                'total_readings': len(readings),
                'average_density': sum(densities) / len(densities),
                'max_density': max(densities),
                'min_density': min(densities),
                'std_density': pd.Series(densities).std(),
                'total_count': sum(counts),
                'average_count': sum(counts) / len(counts),
                'cameras_active': len(set(r.camera_id for r in readings)),
                'time_period_hours': hours,
                'latest_reading': max(r.timestamp for r in readings)
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {e}")
            return {}
        finally:
            session.close()
    
    def get_camera_performance(self) -> Dict:
        """Get performance metrics for each camera"""
        session = self.db_manager.get_session()
        try:
            cameras = session.query(Camera).all()
            performance_data = {}
            
            for camera in cameras:
                # Get recent readings for this camera
                recent_readings = session.query(DensityReading).filter(
                    DensityReading.camera_id == camera.id,
                    DensityReading.timestamp >= datetime.now() - timedelta(hours=24)
                ).all()
                
                if recent_readings:
                    densities = [r.density_value for r in recent_readings]
                    processing_times = [r.processing_time_ms for r in recent_readings if r.processing_time_ms]
                    
                    performance_data[camera.id] = {
                        'name': camera.name,
                        'location': camera.location,
                        'total_readings': len(recent_readings),
                        'average_density': sum(densities) / len(densities),
                        'max_density': max(densities),
                        'average_processing_time': sum(processing_times) / len(processing_times) if processing_times else 0,
                        'last_reading': max(r.timestamp for r in recent_readings),
                        'status': camera.status,
                        'uptime_percentage': self._calculate_uptime(recent_readings, 24)
                    }
                else:
                    performance_data[camera.id] = {
                        'name': camera.name,
                        'location': camera.location,
                        'total_readings': 0,
                        'status': 'inactive',
                        'uptime_percentage': 0
                    }
            
            return performance_data
            
        except Exception as e:
            self.logger.error(f"Error getting camera performance: {e}")
            return {}
        finally:
            session.close()
    
    def _calculate_uptime(self, readings: List, hours: int) -> float:
        """Calculate camera uptime percentage"""
        if not readings:
            return 0.0
        
        # Expected readings (assuming 1 reading per minute)
        expected_readings = hours * 60
        actual_readings = len(readings)
        
        uptime_percentage = min(100.0, (actual_readings / expected_readings) * 100)
        return round(uptime_percentage, 2)
    
    def add_system_log(self, level: str, component: str, message: str, 
                      camera_id: str = None, additional_data: Dict = None):
        """Add system log entry"""
        session = self.db_manager.get_session()
        try:
            log_entry = SystemLog(
                level=level,
                component=component,
                message=message,
                camera_id=camera_id,
                additional_data=json.dumps(additional_data) if additional_data else None,
                timestamp=datetime.now()
            )
            
            session.add(log_entry)
            session.commit()
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error adding system log: {e}")
        finally:
            session.close()
    
    def get_system_logs(self, hours: int = 24, level: str = None, 
                       component: str = None) -> pd.DataFrame:
        """Get system logs"""
        session = self.db_manager.get_session()
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            query = session.query(SystemLog).filter(
                SystemLog.timestamp >= cutoff_time
            )
            
            if level:
                query = query.filter(SystemLog.level == level)
            
            if component:
                query = query.filter(SystemLog.component == component)
            
            query = query.order_by(desc(SystemLog.timestamp))
            
            logs = query.all()
            
            if not logs:
                return pd.DataFrame()
            
            data = []
            for log in logs:
                data.append({
                    'timestamp': log.timestamp,
                    'level': log.level,
                    'component': log.component,
                    'message': log.message,
                    'camera_id': log.camera_id
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            self.logger.error(f"Error getting system logs: {e}")
            return pd.DataFrame()
        finally:
            session.close()
    
    def get_configuration(self, key: str = None, category: str = None) -> Dict:
        """Get configuration values"""
        session = self.db_manager.get_session()
        try:
            query = session.query(Configuration)
            
            if key:
                query = query.filter(Configuration.key == key)
            
            if category:
                query = query.filter(Configuration.category == category)
            
            configs = query.all()
            
            result = {}
            for config in configs:
                # Convert value based on data type
                value = config.value
                if config.data_type == 'float':
                    value = float(value)
                elif config.data_type == 'integer':
                    value = int(value)
                elif config.data_type == 'boolean':
                    value = value.lower() == 'true'
                elif config.data_type == 'json':
                    value = json.loads(value)
                
                result[config.key] = value
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting configuration: {e}")
            return {}
        finally:
            session.close()
    
    def update_configuration(self, key: str, value: Any, updated_by: str = "system") -> bool:
        """Update configuration value"""
        session = self.db_manager.get_session()
        try:
            config = session.query(Configuration).filter(Configuration.key == key).first()
            
            if config:
                # Convert value to string for storage
                if config.data_type == 'json':
                    config.value = json.dumps(value)
                else:
                    config.value = str(value)
                
                config.updated_by = updated_by
                config.updated_at = datetime.now()
                
                session.commit()
                return True
            
            return False
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error updating configuration: {e}")
            return False
        finally:
            session.close()
    
    def get_database_info(self) -> Dict:
        """Get database information and statistics"""
        session = self.db_manager.get_session()
        try:
            # Get table row counts
            table_counts = {}
            
            tables = [
                ('cameras', Camera),
                ('density_readings', DensityReading),
                ('alert_events', AlertEvent),
                ('system_logs', SystemLog),
                ('configurations', Configuration)
            ]
            
            for table_name, model_class in tables:
                count = session.query(model_class).count()
                table_counts[table_name] = count
            
            # Get database size (PostgreSQL specific)
            try:
                db_name = session.get_bind().url.database
                size_query = text(f"SELECT pg_size_pretty(pg_database_size('{db_name}'))")
                result = session.execute(size_query).fetchone()
                db_size = result[0] if result else "Unknown"
            except:
                db_size = "Unknown"
            
            return {
                'database_type': 'PostgreSQL',
                'database_size': db_size,
                'table_counts': table_counts,
                'total_records': sum(table_counts.values()),
                'connection_url': str(session.get_bind().url).replace(session.get_bind().url.password, '***')
            }
            
        except Exception as e:
            self.logger.error(f"Error getting database info: {e}")
            return {}
        finally:
            session.close()
    
    def export_data(self, hours: int = 24, format: str = 'csv', 
                   tables: List[str] = None) -> str:
        """Export data in specified format"""
        try:
            if not tables:
                tables = ['density_readings', 'alert_events']
            
            exported_data = {}
            
            if 'density_readings' in tables:
                density_df = self.get_recent_density_data(hours=hours)
                exported_data['density_readings'] = density_df
            
            if 'alert_events' in tables:
                alerts_df = self.get_alert_events(hours=hours)
                exported_data['alert_events'] = alerts_df
            
            if format.lower() == 'csv':
                # Combine all dataframes with table prefixes
                combined_data = []
                for table_name, df in exported_data.items():
                    if not df.empty:
                        df_copy = df.copy()
                        df_copy['table_source'] = table_name
                        combined_data.append(df_copy)
                
                if combined_data:
                    result_df = pd.concat(combined_data, ignore_index=True)
                    return result_df.to_csv(index=False)
                else:
                    return "No data available for export"
            
            elif format.lower() == 'json':
                result = {}
                for table_name, df in exported_data.items():
                    if not df.empty:
                        result[table_name] = df.to_dict('records')
                
                return json.dumps(result, indent=2, default=str)
            
            else:
                return f"Unsupported format: {format}"
                
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return f"Error exporting data: {e}"
    
    def clear_old_data(self, days: int = 30):
        """Clear old data from database"""
        session = self.db_manager.get_session()
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Clear old density readings
            deleted_readings = session.query(DensityReading).filter(
                DensityReading.timestamp < cutoff_date
            ).delete()
            
            # Clear old alert events
            deleted_alerts = session.query(AlertEvent).filter(
                AlertEvent.created_at < cutoff_date
            ).delete()
            
            # Clear old system logs
            deleted_logs = session.query(SystemLog).filter(
                SystemLog.timestamp < cutoff_date
            ).delete()
            
            session.commit()
            
            self.add_system_log("INFO", "DATA_SERVICE", 
                              f"Cleaned up old data: {deleted_readings} readings, {deleted_alerts} alerts, {deleted_logs} logs")
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error clearing old data: {e}")
        finally:
            session.close()