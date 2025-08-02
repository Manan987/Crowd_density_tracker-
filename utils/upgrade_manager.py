"""
Upgrade Manager for CrowdGuard Pro
Handles application upgrades, migrations, and rollbacks
"""

import os
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
from dataclasses import dataclass, asdict

from utils.version_manager import VersionManager, Version

@dataclass
class UpgradeStep:
    """Represents a single upgrade step"""
    name: str
    description: str
    function: str
    required: bool = True
    rollback_function: Optional[str] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class UpgradeResult:
    """Result of an upgrade operation"""
    success: bool
    version_from: str
    version_to: str
    steps_completed: List[str]
    steps_failed: List[str]
    errors: List[str]
    rollback_available: bool
    duration: float
    timestamp: str

class UpgradeManager:
    """Manages application upgrades and migrations"""
    
    def __init__(self, app_dir: str = "."):
        self.app_dir = Path(app_dir)
        self.version_manager = VersionManager()
        self.backup_dir = self.app_dir / "backups"
        self.upgrade_log = self.app_dir / "logs" / "upgrades.log"
        self.upgrade_config = self.app_dir / "upgrade_config.json"
        
        # Ensure directories exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.upgrade_log.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            filename=self.upgrade_log,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load upgrade configuration
        self.upgrade_steps = self._load_upgrade_steps()
    
    def _load_upgrade_steps(self) -> Dict[str, List[UpgradeStep]]:
        """Load upgrade steps configuration"""
        default_steps = {
            "1.0.0": [
                UpgradeStep(
                    name="backup_database",
                    description="Backup current database",
                    function="backup_database",
                    required=True,
                    rollback_function="restore_database"
                ),
                UpgradeStep(
                    name="update_dependencies",
                    description="Update Python dependencies",
                    function="update_dependencies",
                    required=True
                ),
                UpgradeStep(
                    name="migrate_database",
                    description="Run database migrations",
                    function="migrate_database",
                    required=True,
                    dependencies=["backup_database"]
                ),
                UpgradeStep(
                    name="update_models",
                    description="Update AI model weights",
                    function="update_models",
                    required=False
                ),
                UpgradeStep(
                    name="clear_cache",
                    description="Clear application cache",
                    function="clear_cache",
                    required=False
                ),
                UpgradeStep(
                    name="update_config",
                    description="Update configuration files",
                    function="update_config",
                    required=True,
                    rollback_function="restore_config"
                )
            ]
        }
        
        try:
            if self.upgrade_config.exists():
                with open(self.upgrade_config, 'r') as f:
                    data = json.load(f)
                    steps = {}
                    for version, step_list in data.items():
                        steps[version] = [UpgradeStep(**step) for step in step_list]
                    return steps
        except Exception as e:
            self.logger.error(f"Failed to load upgrade config: {e}")
        
        return default_steps
    
    def create_backup(self, backup_name: Optional[str] = None) -> str:
        """Create a full backup of the application"""
        if backup_name is None:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)
        
        self.logger.info(f"Creating backup: {backup_name}")
        
        # Backup critical files and directories
        critical_paths = [
            "app.py",
            "models/",
            "utils/",
            "config/",
            "database/",
            "pyproject.toml",
            ".env",
            "version.json"
        ]
        
        for path in critical_paths:
            src = self.app_dir / path
            if src.exists():
                dst = backup_path / path
                if src.is_file():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                else:
                    shutil.copytree(src, dst, dirs_exist_ok=True)
        
        # Save backup metadata
        metadata = {
            "backup_name": backup_name,
            "timestamp": datetime.now().isoformat(),
            "version": str(self.version_manager.current_version),
            "files_backed_up": [str(p) for p in critical_paths if (self.app_dir / p).exists()]
        }
        
        with open(backup_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Backup completed: {backup_path}")
        return str(backup_path)
    
    def restore_backup(self, backup_name: str) -> bool:
        """Restore from a backup"""
        backup_path = self.backup_dir / backup_name
        
        if not backup_path.exists():
            self.logger.error(f"Backup not found: {backup_name}")
            return False
        
        try:
            self.logger.info(f"Restoring from backup: {backup_name}")
            
            # Load backup metadata
            metadata_file = backup_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.logger.info(f"Restoring backup from {metadata['timestamp']}")
            
            # Restore files
            for item in backup_path.iterdir():
                if item.name == "metadata.json":
                    continue
                
                dst = self.app_dir / item.name
                if item.is_file():
                    shutil.copy2(item, dst)
                else:
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(item, dst)
            
            self.logger.info("Backup restoration completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore backup: {e}")
            return False
    
    def backup_database(self) -> bool:
        """Backup the database"""
        try:
            # Implementation depends on database type
            db_backup_path = self.backup_dir / f"db_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
            
            # For SQLite
            if "sqlite" in os.getenv("DATABASE_URL", ""):
                db_path = os.getenv("DATABASE_URL", "").replace("sqlite:///", "")
                if os.path.exists(db_path):
                    shutil.copy2(db_path, db_backup_path.with_suffix('.db'))
            
            # For PostgreSQL
            elif "postgresql" in os.getenv("DATABASE_URL", ""):
                subprocess.run([
                    "pg_dump", os.getenv("DATABASE_URL", ""),
                    "-f", str(db_backup_path)
                ], check=True)
            
            self.logger.info(f"Database backup created: {db_backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Database backup failed: {e}")
            return False
    
    def update_dependencies(self) -> bool:
        """Update Python dependencies"""
        try:
            self.logger.info("Updating dependencies...")
            
            # Use uv if available, otherwise pip
            if shutil.which("uv"):
                subprocess.run(["uv", "pip", "install", "-e", "."], check=True)
            else:
                subprocess.run(["pip", "install", "-e", "."], check=True)
            
            self.logger.info("Dependencies updated successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to update dependencies: {e}")
            return False
    
    def migrate_database(self) -> bool:
        """Run database migrations"""
        try:
            self.logger.info("Running database migrations...")
            
            # Run Alembic migrations if available
            if (self.app_dir / "alembic.ini").exists():
                subprocess.run(["alembic", "upgrade", "head"], check=True)
            
            self.logger.info("Database migrations completed")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Database migration failed: {e}")
            return False
    
    def update_models(self) -> bool:
        """Update AI model weights"""
        try:
            self.logger.info("Checking for model updates...")
            
            # This would typically download new model weights
            # For now, just log the action
            models_dir = self.app_dir / "models" / "weights"
            if models_dir.exists():
                self.logger.info("Model weights directory exists")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model update failed: {e}")
            return False
    
    def clear_cache(self) -> bool:
        """Clear application cache"""
        try:
            cache_dirs = [
                self.app_dir / "temp",
                self.app_dir / "__pycache__",
                self.app_dir / ".streamlit"
            ]
            
            for cache_dir in cache_dirs:
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
                    self.logger.info(f"Cleared cache: {cache_dir}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Cache clearing failed: {e}")
            return False
    
    def update_config(self) -> bool:
        """Update configuration files"""
        try:
            self.logger.info("Updating configuration...")
            
            # Backup current config
            config_backup = self.backup_dir / "config_backup.json"
            
            # Update would happen here
            # For now, just log the action
            
            self.logger.info("Configuration updated")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration update failed: {e}")
            return False
    
    def perform_upgrade(self, target_version: str, force: bool = False) -> UpgradeResult:
        """Perform upgrade to target version"""
        start_time = datetime.now()
        current_version = str(self.version_manager.current_version)
        
        result = UpgradeResult(
            success=False,
            version_from=current_version,
            version_to=target_version,
            steps_completed=[],
            steps_failed=[],
            errors=[],
            rollback_available=False,
            duration=0.0,
            timestamp=start_time.isoformat()
        )
        
        try:
            # Validate upgrade
            if not force:
                is_valid, issues = self.version_manager.validate_upgrade(target_version)
                if not is_valid:
                    result.errors.extend(issues)
                    return result
            
            # Create backup
            backup_name = f"pre_upgrade_{current_version}_to_{target_version}_{start_time.strftime('%Y%m%d_%H%M%S')}"
            self.create_backup(backup_name)
            result.rollback_available = True
            
            # Get upgrade steps
            steps = self.upgrade_steps.get(target_version, [])
            
            # Execute upgrade steps
            for step in steps:
                try:
                    self.logger.info(f"Executing step: {step.name}")
                    
                    # Get the function to execute
                    func = getattr(self, step.function, None)
                    if func and callable(func):
                        success = func()
                        if success:
                            result.steps_completed.append(step.name)
                            self.logger.info(f"Step completed: {step.name}")
                        else:
                            result.steps_failed.append(step.name)
                            if step.required:
                                raise Exception(f"Required step failed: {step.name}")
                    else:
                        self.logger.warning(f"Function not found: {step.function}")
                        
                except Exception as e:
                    result.steps_failed.append(step.name)
                    result.errors.append(str(e))
                    self.logger.error(f"Step failed: {step.name} - {e}")
                    
                    if step.required:
                        # Attempt rollback
                        self.logger.info("Required step failed, attempting rollback...")
                        self.restore_backup(backup_name)
                        return result
            
            # Update version information
            self.version_manager.current_version = Version.from_string(target_version)
            self.version_manager.save_version_info()
            
            result.success = True
            self.logger.info(f"Upgrade completed successfully: {current_version} -> {target_version}")
            
        except Exception as e:
            result.errors.append(str(e))
            self.logger.error(f"Upgrade failed: {e}")
        
        finally:
            end_time = datetime.now()
            result.duration = (end_time - start_time).total_seconds()
        
        return result
    
    def get_available_upgrades(self) -> List[str]:
        """Get list of available upgrade versions"""
        current = self.version_manager.current_version
        available = []
        
        for version_str in self.upgrade_steps.keys():
            version = Version.from_string(version_str)
            if version > current:
                available.append(version_str)
        
        return sorted(available)
    
    def get_upgrade_history(self) -> List[Dict]:
        """Get history of previous upgrades"""
        history = []
        
        # This would typically read from a persistent log
        # For now, return empty list
        
        return history

# Global upgrade manager instance
upgrade_manager = UpgradeManager()

def get_upgrade_manager() -> UpgradeManager:
    """Get upgrade manager instance"""
    return upgrade_manager
