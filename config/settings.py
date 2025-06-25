import json
import os
from typing import Dict, Any

class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            "model": {
                "type": "csrnet",
                "input_size": [512, 512],
                "device": "auto"
            },
            "processing": {
                "max_streams": 4,
                "buffer_size": 10,
                "fps_target": 30
            },
            "alerts": {
                "density_high": 0.7,
                "density_critical": 0.9,
                "count_high": 100,
                "count_critical": 200,
                "rapid_increase_rate": 0.5,
                "sustained_duration": 60
            },
            "notifications": {
                "email_enabled": False,
                "webhook_enabled": False,
                "sms_enabled": False
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    default_config.update(loaded_config)
            except Exception as e:
                print(f"Error loading config: {e}")
        
        return default_config
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self.save_config()