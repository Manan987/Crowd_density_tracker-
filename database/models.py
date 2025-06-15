from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
import os
from datetime import datetime

Base = declarative_base()

class Camera(Base):
    """Camera configuration and metadata"""
    __tablename__ = 'cameras'
    
    id = Column(String(50), primary_key=True)
    name = Column(String(200), nullable=False)
    location = Column(String(500))
    resolution_width = Column(Integer, default=1920)
    resolution_height = Column(Integer, default=1080)
    fps = Column(Integer, default=30)
    area_m2 = Column(Float, default=100.0)
    status = Column(String(20), default='active')
    calibration_data = Column(Text)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    density_readings = relationship("DensityReading", back_populates="camera")
    alert_events = relationship("AlertEvent", back_populates="camera")

class DensityReading(Base):
    """Individual crowd density measurements"""
    __tablename__ = 'density_readings'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(50), ForeignKey('cameras.id'), nullable=False)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    density_value = Column(Float, nullable=False)  # people per m²
    crowd_count = Column(Integer, nullable=False)
    confidence_score = Column(Float, default=0.0)
    processing_time_ms = Column(Float)
    frame_quality_score = Column(Float)
    weather_condition = Column(String(50))
    ambient_light_level = Column(Float)
    
    # Relationships
    camera = relationship("Camera", back_populates="density_readings")

class AlertEvent(Base):
    """Alert events and notifications"""
    __tablename__ = 'alert_events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(50), ForeignKey('cameras.id'), nullable=False)
    alert_type = Column(String(20), nullable=False)  # Normal, Warning, Critical
    density_value = Column(Float, nullable=False)
    threshold_exceeded = Column(Float)
    duration_seconds = Column(Integer)
    acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String(100))
    acknowledged_at = Column(DateTime)
    resolved_at = Column(DateTime)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    camera = relationship("Camera", back_populates="alert_events")
    notifications = relationship("NotificationLog", back_populates="alert_event")

class NotificationLog(Base):
    """Log of sent notifications"""
    __tablename__ = 'notification_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_event_id = Column(Integer, ForeignKey('alert_events.id'))
    notification_type = Column(String(50))  # email, sms, push, webhook
    recipient = Column(String(200))
    status = Column(String(20))  # sent, failed, pending
    sent_at = Column(DateTime, default=func.now())
    error_message = Column(Text)
    
    # Relationships
    alert_event = relationship("AlertEvent", back_populates="notifications")

class SystemLog(Base):
    """System operation logs"""
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=func.now())
    level = Column(String(20))  # INFO, WARNING, ERROR, CRITICAL
    component = Column(String(100))  # AI_MODEL, VIDEO_PROCESSOR, ALERT_SYSTEM
    message = Column(Text)
    camera_id = Column(String(50), ForeignKey('cameras.id'))
    user_id = Column(String(100))
    session_id = Column(String(100))
    additional_data = Column(Text)  # JSON data

class ModelPerformance(Base):
    """AI model performance metrics"""
    __tablename__ = 'model_performance'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=func.now())
    model_version = Column(String(50))
    accuracy_score = Column(Float)
    precision_score = Column(Float)
    recall_score = Column(Float)
    f1_score = Column(Float)
    inference_time_ms = Column(Float)
    memory_usage_mb = Column(Float)
    gpu_utilization = Column(Float)
    total_predictions = Column(Integer)
    correct_predictions = Column(Integer)

class UserSession(Base):
    """User session tracking"""
    __tablename__ = 'user_sessions'
    
    id = Column(String(100), primary_key=True)
    user_id = Column(String(100))
    start_time = Column(DateTime, default=func.now())
    end_time = Column(DateTime)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    actions_count = Column(Integer, default=0)
    pages_visited = Column(Text)  # JSON array
    alerts_acknowledged = Column(Integer, default=0)

class Configuration(Base):
    """System configuration parameters"""
    __tablename__ = 'configurations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(Text)
    data_type = Column(String(20))  # string, integer, float, boolean, json
    category = Column(String(50))  # alert_thresholds, notification_settings, model_config
    description = Column(Text)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    updated_by = Column(String(100))

class DatabaseManager:
    """Database connection and session management"""
    
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self.engine = create_engine(
            self.database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False
        )
        
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def initialize_default_data(self):
        """Initialize default cameras and configuration"""
        session = self.get_session()
        try:
            # Check if cameras exist
            if session.query(Camera).count() == 0:
                default_cameras = [
                    Camera(
                        id="CAM_001",
                        name="Main Entrance",
                        location="Building entrance, facing inward",
                        area_m2=150.0,
                        status="active"
                    ),
                    Camera(
                        id="CAM_002", 
                        name="Central Plaza",
                        location="Central courtyard area",
                        area_m2=500.0,
                        status="active"
                    ),
                    Camera(
                        id="CAM_003",
                        name="Exit Gate",
                        location="Main exit point",
                        area_m2=120.0,
                        status="active"
                    ),
                    Camera(
                        id="CAM_004",
                        name="Emergency Route",
                        location="Emergency evacuation pathway",
                        area_m2=80.0,
                        status="active"
                    )
                ]
                
                for camera in default_cameras:
                    session.add(camera)
            
            # Initialize default configuration
            if session.query(Configuration).count() == 0:
                default_configs = [
                    Configuration(
                        key="alert_warning_threshold",
                        value="2.0",
                        data_type="float",
                        category="alert_thresholds",
                        description="Density threshold for warning alerts (people/m²)"
                    ),
                    Configuration(
                        key="alert_critical_threshold", 
                        value="4.0",
                        data_type="float",
                        category="alert_thresholds",
                        description="Density threshold for critical alerts (people/m²)"
                    ),
                    Configuration(
                        key="alert_duration_seconds",
                        value="30",
                        data_type="integer", 
                        category="alert_thresholds",
                        description="Minimum time between consecutive alerts"
                    ),
                    Configuration(
                        key="email_notifications_enabled",
                        value="true",
                        data_type="boolean",
                        category="notification_settings",
                        description="Enable email notifications"
                    ),
                    Configuration(
                        key="model_confidence_threshold",
                        value="0.85",
                        data_type="float",
                        category="model_config", 
                        description="Minimum confidence score for density predictions"
                    )
                ]
                
                for config in default_configs:
                    session.add(config)
            
            session.commit()
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()