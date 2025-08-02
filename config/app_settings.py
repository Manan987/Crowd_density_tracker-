from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, List
import os
from pathlib import Path

class DatabaseSettings(BaseSettings):
    """Database configuration settings"""
    url: str = Field(default="sqlite:///crowd_monitoring.db", env="DATABASE_URL")
    pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    echo: bool = Field(default=False, env="DATABASE_ECHO")

class RedisSettings(BaseSettings):
    """Redis configuration settings"""
    url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    max_connections: int = Field(default=10, env="REDIS_MAX_CONNECTIONS")

class ModelSettings(BaseSettings):
    """AI Model configuration settings"""
    path: str = Field(default="./models/weights/", env="MODEL_PATH")
    confidence_threshold: float = Field(default=0.5, env="MODEL_CONFIDENCE_THRESHOLD")
    nms_threshold: float = Field(default=0.4, env="MODEL_NMS_THRESHOLD")
    enable_gpu: bool = Field(default=True, env="ENABLE_GPU")
    batch_size: int = Field(default=4, env="BATCH_SIZE")
    device: str = Field(default="auto", env="MODEL_DEVICE")

class VideoSettings(BaseSettings):
    """Video processing configuration"""
    max_concurrent_streams: int = Field(default=5, env="MAX_CONCURRENT_STREAMS")
    frame_skip: int = Field(default=2, env="FRAME_SKIP")
    buffer_size: int = Field(default=30, env="VIDEO_BUFFER_SIZE")
    supported_formats: List[str] = Field(default=["mp4", "avi", "mov", "mkv"])

class AlertSettings(BaseSettings):
    """Alert system configuration"""
    enable_email: bool = Field(default=True, env="ENABLE_EMAIL_ALERTS")
    email_smtp_server: str = Field(default="smtp.gmail.com", env="EMAIL_SMTP_SERVER")
    email_smtp_port: int = Field(default=587, env="EMAIL_SMTP_PORT")
    email_username: Optional[str] = Field(default=None, env="EMAIL_USERNAME")
    email_password: Optional[str] = Field(default=None, env="EMAIL_PASSWORD")
    email_from: Optional[str] = Field(default=None, env="EMAIL_FROM")
    
    enable_sms: bool = Field(default=False, env="ENABLE_SMS_ALERTS")
    sms_api_key: Optional[str] = Field(default=None, env="SMS_API_KEY")
    sms_api_secret: Optional[str] = Field(default=None, env="SMS_API_SECRET")
    
    enable_slack: bool = Field(default=False, env="ENABLE_SLACK_ALERTS")
    slack_webhook_url: Optional[str] = Field(default=None, env="SLACK_WEBHOOK_URL")

class MonitoringSettings(BaseSettings):
    """Monitoring and metrics configuration"""
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    prometheus_endpoint: str = Field(default="/metrics", env="PROMETHEUS_ENDPOINT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

class SecuritySettings(BaseSettings):
    """Security configuration"""
    secret_key: str = Field(default="dev-secret-key", env="SECRET_KEY")
    api_key_required: bool = Field(default=False, env="API_KEY_REQUIRED")
    api_rate_limit: int = Field(default=100, env="API_RATE_LIMIT")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")

class StorageSettings(BaseSettings):
    """File storage configuration"""
    upload_folder: str = Field(default="./uploads", env="UPLOAD_FOLDER")
    temp_folder: str = Field(default="./temp", env="TEMP_FOLDER")
    logs_folder: str = Field(default="./logs", env="LOGS_FOLDER")
    max_upload_size: int = Field(default=200, env="MAX_UPLOAD_SIZE")  # MB

class PerformanceSettings(BaseSettings):
    """Performance tuning configuration"""
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    cache_ttl: int = Field(default=300, env="CACHE_TTL")  # seconds
    worker_processes: int = Field(default=4, env="WORKER_PROCESSES")
    worker_connections: int = Field(default=1000, env="WORKER_CONNECTIONS")

class AppSettings(BaseSettings):
    """Main application configuration"""
    name: str = Field(default="CrowdGuard Pro", env="APP_NAME")
    version: str = Field(default="1.0.0", env="APP_VERSION")
    environment: str = Field(default="development", env="APP_ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8501, env="PORT")
    
    # Sub-configurations
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    model: ModelSettings = ModelSettings()
    video: VideoSettings = VideoSettings()
    alerts: AlertSettings = AlertSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    security: SecuritySettings = SecuritySettings()
    storage: StorageSettings = StorageSettings()
    performance: PerformanceSettings = PerformanceSettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.storage.upload_folder,
            self.storage.temp_folder,
            self.storage.logs_folder,
            self.model.path,
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

# Global settings instance
settings = AppSettings()

def get_settings() -> AppSettings:
    """Get application settings instance"""
    return settings

def reload_settings():
    """Reload settings from environment"""
    global settings
    settings = AppSettings()
    return settings
