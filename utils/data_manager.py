import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import sqlite3
import os
import threading
import json

class DataManager:
    """
    Data management system for crowd density monitoring
    Handles data storage, retrieval, and analysis
    """
    
    def __init__(self, db_path: str = "crowd_monitoring.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
        
        # In-memory cache for recent data
        self.recent_data = []
        self.max_cache_size = 1000
        
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Density data table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS density_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        camera_id TEXT NOT NULL,
                        density REAL NOT NULL,
                        crowd_count INTEGER NOT NULL,
                        alert_level TEXT NOT NULL,
                        processed_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Alert events table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS alert_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        camera_id TEXT NOT NULL,
                        alert_level TEXT NOT NULL,
                        density REAL NOT NULL,
                        acknowledged BOOLEAN DEFAULT FALSE,
                        acknowledged_by TEXT,
                        acknowledged_at DATETIME,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # System logs table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        level TEXT NOT NULL,
                        message TEXT NOT NULL,
                        camera_id TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for better query performance
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_density_timestamp 
                    ON density_data(timestamp)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_density_camera 
                    ON density_data(camera_id)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_alerts_timestamp 
                    ON alert_events(timestamp)
                ''')
                
                conn.commit()
                print("Database initialized successfully")
                
        except Exception as e:
            print(f"Error initializing database: {e}")
    
    def add_data_point(self, camera_id: str, density: float, 
                      crowd_count: int, alert_level: str) -> bool:
        """
        Add a new data point to the system
        
        Args:
            camera_id: Camera identifier
            density: Crowd density value
            crowd_count: Estimated crowd count
            alert_level: Current alert level
            
        Returns:
            Success status
        """
        try:
            timestamp = datetime.now()
            
            # Add to in-memory cache
            data_point = {
                'timestamp': timestamp,
                'camera_id': camera_id,
                'density': density,
                'crowd_count': crowd_count,
                'alert_level': alert_level
            }
            
            with self.lock:
                self.recent_data.append(data_point)
                
                # Maintain cache size
                if len(self.recent_data) > self.max_cache_size:
                    self.recent_data = self.recent_data[-self.max_cache_size:]
            
            # Add to database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO density_data 
                    (timestamp, camera_id, density, crowd_count, alert_level)
                    VALUES (?, ?, ?, ?, ?)
                ''', (timestamp, camera_id, density, crowd_count, alert_level))
                
                # Add alert event if not normal
                if alert_level != "Normal":
                    cursor.execute('''
                        INSERT INTO alert_events 
                        (timestamp, camera_id, alert_level, density)
                        VALUES (?, ?, ?, ?)
                    ''', (timestamp, camera_id, alert_level, density))
                
                conn.commit()
            
            return True
            
        except Exception as e:
            print(f"Error adding data point: {e}")
            return False
    
    def get_recent_data(self, minutes: int = 60, hours: int = None, 
                       cameras: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get recent data from cache or database
        
        Args:
            minutes: Minutes to look back (ignored if hours is provided)
            hours: Hours to look back
            cameras: List of camera IDs to filter by
            
        Returns:
            DataFrame with recent data
        """
        try:
            # Calculate time cutoff
            if hours is not None:
                cutoff_time = datetime.now() - timedelta(hours=hours)
            else:
                cutoff_time = datetime.now() - timedelta(minutes=minutes)
            
            # Try cache first for recent data (last hour)
            if hours is None and minutes <= 60:
                with self.lock:
                    cache_data = [
                        dp for dp in self.recent_data 
                        if dp['timestamp'] >= cutoff_time
                    ]
                
                if cache_data:
                    df = pd.DataFrame(cache_data)
                    
                    # Filter by cameras if specified
                    if cameras:
                        df = df[df['camera_id'].isin(cameras)]
                    
                    return df
            
            # Query database for older data
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT timestamp, camera_id, density, crowd_count, alert_level
                    FROM density_data 
                    WHERE timestamp >= ?
                '''
                params = [cutoff_time]
                
                if cameras:
                    placeholders = ','.join(['?' for _ in cameras])
                    query += f' AND camera_id IN ({placeholders})'
                    params.extend(cameras)
                
                query += ' ORDER BY timestamp ASC'
                
                df = pd.read_sql_query(query, conn, params=params)
                
                # Convert timestamp to datetime
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                return df
            
        except Exception as e:
            print(f"Error getting recent data: {e}")
            return pd.DataFrame()
    
    def get_alert_events(self, hours: int = 24) -> pd.DataFrame:
        """
        Get alert events from specified time period
        
        Args:
            hours: Hours to look back
            
        Returns:
            DataFrame with alert events
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT timestamp, camera_id, alert_level, density, 
                           acknowledged, acknowledged_by, acknowledged_at
                    FROM alert_events 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                '''
                
                df = pd.read_sql_query(query, conn, params=[cutoff_time])
                
                # Convert timestamps to datetime
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['acknowledged_at'] = pd.to_datetime(df['acknowledged_at'])
                
                return df
            
        except Exception as e:
            print(f"Error getting alert events: {e}")
            return pd.DataFrame()
    
    def get_density_statistics(self, hours: int = 24) -> Dict:
        """
        Get density statistics for specified time period
        
        Args:
            hours: Hours to look back
            
        Returns:
            Dictionary with statistics
        """
        try:
            data = self.get_recent_data(hours=hours)
            
            if data.empty:
                return {}
            
            stats = {
                'total_readings': len(data),
                'average_density': data['density'].mean(),
                'max_density': data['density'].max(),
                'min_density': data['density'].min(),
                'std_density': data['density'].std(),
                'total_count': data['crowd_count'].sum(),
                'average_count': data['crowd_count'].mean(),
                'cameras_active': len(data['camera_id'].unique()),
                'alert_distribution': data['alert_level'].value_counts().to_dict(),
                'time_period_hours': hours
            }
            
            return stats
            
        except Exception as e:
            print(f"Error calculating statistics: {e}")
            return {}
    
    def get_camera_performance(self) -> Dict:
        """Get performance metrics for each camera"""
        try:
            data = self.get_recent_data(hours=24)
            
            if data.empty:
                return {}
            
            camera_stats = {}
            
            for camera_id in data['camera_id'].unique():
                camera_data = data[data['camera_id'] == camera_id]
                
                camera_stats[camera_id] = {
                    'total_readings': len(camera_data),
                    'average_density': camera_data['density'].mean(),
                    'max_density': camera_data['density'].max(),
                    'alert_count': len(camera_data[camera_data['alert_level'] != 'Normal']),
                    'last_reading': camera_data['timestamp'].max(),
                    'uptime_percentage': self._calculate_uptime(camera_data)
                }
            
            return camera_stats
            
        except Exception as e:
            print(f"Error getting camera performance: {e}")
            return {}
    
    def _calculate_uptime(self, camera_data: pd.DataFrame) -> float:
        """Calculate camera uptime percentage"""
        try:
            if len(camera_data) < 2:
                return 100.0
            
            # Calculate expected readings (assuming 1 reading per minute)
            time_span = (camera_data['timestamp'].max() - 
                        camera_data['timestamp'].min()).total_seconds()
            expected_readings = time_span / 60  # 1 reading per minute
            
            actual_readings = len(camera_data)
            uptime_percentage = min(100.0, (actual_readings / expected_readings) * 100)
            
            return round(uptime_percentage, 2)
            
        except Exception as e:
            print(f"Error calculating uptime: {e}")
            return 0.0
    
    def export_data(self, hours: int = 24, format: str = 'csv') -> str:
        """
        Export data to specified format
        
        Args:
            hours: Hours of data to export
            format: Export format ('csv' or 'json')
            
        Returns:
            Exported data as string
        """
        try:
            data = self.get_recent_data(hours=hours)
            
            if data.empty:
                return "No data available for export"
            
            if format.lower() == 'csv':
                return data.to_csv(index=False)
            elif format.lower() == 'json':
                # Convert datetime to string for JSON serialization
                data['timestamp'] = data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                return data.to_json(orient='records', indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
        except Exception as e:
            print(f"Error exporting data: {e}")
            return f"Error exporting data: {e}"
    
    def add_system_log(self, level: str, message: str, camera_id: Optional[str] = None):
        """
        Add system log entry
        
        Args:
            level: Log level (INFO, WARNING, ERROR, CRITICAL)
            message: Log message
            camera_id: Optional camera ID
        """
        try:
            timestamp = datetime.now()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO system_logs (timestamp, level, message, camera_id)
                    VALUES (?, ?, ?, ?)
                ''', (timestamp, level, message, camera_id))
                conn.commit()
                
        except Exception as e:
            print(f"Error adding system log: {e}")
    
    def get_system_logs(self, hours: int = 24, level: Optional[str] = None) -> pd.DataFrame:
        """
        Get system logs
        
        Args:
            hours: Hours to look back
            level: Optional log level filter
            
        Returns:
            DataFrame with system logs
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT timestamp, level, message, camera_id
                    FROM system_logs 
                    WHERE timestamp >= ?
                '''
                params = [cutoff_time]
                
                if level:
                    query += ' AND level = ?'
                    params.append(level)
                
                query += ' ORDER BY timestamp DESC'
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                return df
            
        except Exception as e:
            print(f"Error getting system logs: {e}")
            return pd.DataFrame()
    
    def clear_data(self, older_than_days: int = None):
        """
        Clear data from database
        
        Args:
            older_than_days: If specified, only clear data older than this many days
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if older_than_days:
                    cutoff_time = datetime.now() - timedelta(days=older_than_days)
                    
                    cursor.execute('DELETE FROM density_data WHERE timestamp < ?', [cutoff_time])
                    cursor.execute('DELETE FROM alert_events WHERE timestamp < ?', [cutoff_time])
                    cursor.execute('DELETE FROM system_logs WHERE timestamp < ?', [cutoff_time])
                    
                    print(f"Cleared data older than {older_than_days} days")
                else:
                    cursor.execute('DELETE FROM density_data')
                    cursor.execute('DELETE FROM alert_events')
                    cursor.execute('DELETE FROM system_logs')
                    
                    print("Cleared all data")
                
                conn.commit()
            
            # Clear in-memory cache
            with self.lock:
                if older_than_days:
                    cutoff_time = datetime.now() - timedelta(days=older_than_days)
                    self.recent_data = [
                        dp for dp in self.recent_data 
                        if dp['timestamp'] >= cutoff_time
                    ]
                else:
                    self.recent_data = []
                
        except Exception as e:
            print(f"Error clearing data: {e}")
    
    def backup_database(self, backup_path: str) -> bool:
        """
        Create database backup
        
        Args:
            backup_path: Path for backup file
            
        Returns:
            Success status
        """
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            print(f"Database backed up to {backup_path}")
            return True
            
        except Exception as e:
            print(f"Error backing up database: {e}")
            return False
    
    def get_database_info(self) -> Dict:
        """Get database information and statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get table sizes
                tables = ['density_data', 'alert_events', 'system_logs']
                table_info = {}
                
                for table in tables:
                    cursor.execute(f'SELECT COUNT(*) FROM {table}')
                    count = cursor.fetchone()[0]
                    table_info[table] = count
                
                # Get database file size
                db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                
                return {
                    'database_path': self.db_path,
                    'database_size_bytes': db_size,
                    'table_counts': table_info,
                    'cache_size': len(self.recent_data),
                    'max_cache_size': self.max_cache_size
                }
                
        except Exception as e:
            print(f"Error getting database info: {e}")
            return {}
