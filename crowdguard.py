#!/usr/bin/env python3
"""
CrowdGuard Pro CLI Management Tool
Provides command-line interface for managing the application
"""

import click
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from utils.version_manager import get_version, get_version_info, version_manager
    from utils.upgrade_manager import get_upgrade_manager
    from config.app_settings import get_settings
except ImportError as e:
    click.echo(f"Error importing modules: {e}", err=True)
    click.echo("Make sure you're running from the project root directory", err=True)
    sys.exit(1)

@click.group()
@click.version_option(version=get_version(), prog_name="CrowdGuard Pro")
def cli():
    """CrowdGuard Pro - AI-powered crowd density monitoring system"""
    pass

@cli.group()
def version():
    """Version management commands"""
    pass

@version.command()
def show():
    """Show current version information"""
    info = get_version_info()
    
    click.echo("🛡️  CrowdGuard Pro Version Information")
    click.echo("=" * 40)
    click.echo(f"Version: {info['version']}")
    click.echo(f"Python: {info['python_version']}")
    click.echo()
    click.echo("Dependencies:")
    for dep, ver in info['dependencies'].items():
        click.echo(f"  {dep}: {ver}")

@version.command()
@click.argument('target_version')
@click.option('--check-only', is_flag=True, help='Only check compatibility, don\'t upgrade')
def upgrade(target_version: str, check_only: bool):
    """Upgrade to target version"""
    upgrade_mgr = get_upgrade_manager()
    
    if check_only:
        is_valid, issues = version_manager.validate_upgrade(target_version)
        if is_valid:
            click.echo(f"✅ Upgrade to {target_version} is possible")
            upgrade_path = version_manager.get_upgrade_path(target_version)
            if upgrade_path:
                click.echo("Upgrade path:")
                for step in upgrade_path:
                    click.echo(f"  → {step}")
        else:
            click.echo(f"❌ Upgrade to {target_version} has issues:")
            for issue in issues:
                click.echo(f"  • {issue}")
        return
    
    click.echo(f"🚀 Starting upgrade to version {target_version}")
    
    if not click.confirm("This will create a backup and upgrade the application. Continue?"):
        click.echo("Upgrade cancelled")
        return
    
    with click.progressbar(length=100, label='Upgrading') as bar:
        result = upgrade_mgr.perform_upgrade(target_version)
        bar.update(100)
    
    if result.success:
        click.echo(f"✅ Upgrade completed successfully!")
        click.echo(f"Version: {result.version_from} → {result.version_to}")
        click.echo(f"Duration: {result.duration:.2f} seconds")
        if result.steps_completed:
            click.echo("Steps completed:")
            for step in result.steps_completed:
                click.echo(f"  ✅ {step}")
    else:
        click.echo(f"❌ Upgrade failed!")
        if result.errors:
            click.echo("Errors:")
            for error in result.errors:
                click.echo(f"  • {error}")
        if result.rollback_available:
            click.echo("A backup was created and can be restored if needed")

@cli.group()
def config():
    """Configuration management commands"""
    pass

@config.command()
def show():
    """Show current configuration"""
    settings = get_settings()
    
    click.echo("⚙️  CrowdGuard Pro Configuration")
    click.echo("=" * 40)
    
    config_dict = {
        "Application": {
            "Name": settings.name,
            "Version": settings.version,
            "Environment": settings.environment,
            "Debug": settings.debug,
            "Host": settings.host,
            "Port": settings.port,
        },
        "Database": {
            "URL": settings.database.url,
            "Pool Size": settings.database.pool_size,
        },
        "Model": {
            "Path": settings.model.path,
            "Confidence Threshold": settings.model.confidence_threshold,
            "GPU Enabled": settings.model.enable_gpu,
        },
        "Video Processing": {
            "Max Streams": settings.video.max_concurrent_streams,
            "Frame Skip": settings.video.frame_skip,
        },
        "Alerts": {
            "Email Enabled": settings.alerts.enable_email,
            "SMS Enabled": settings.alerts.enable_sms,
            "Slack Enabled": settings.alerts.enable_slack,
        }
    }
    
    for section, values in config_dict.items():
        click.echo(f"\n{section}:")
        for key, value in values.items():
            click.echo(f"  {key}: {value}")

@config.command()
@click.argument('key')
@click.argument('value')
def set(key: str, value: str):
    """Set configuration value"""
    # This would update the configuration
    # For now, just show what would be updated
    click.echo(f"Would set {key} = {value}")
    click.echo("Note: Configuration updates will be implemented in future version")

@cli.group()
def database():
    """Database management commands"""
    pass

@database.command()
def init():
    """Initialize database"""
    click.echo("🗃️  Initializing database...")
    
    try:
        # Import here to avoid circular imports
        from database.models import Base
        from sqlalchemy import create_engine
        from config.app_settings import get_settings
        
        settings = get_settings()
        engine = create_engine(settings.database.url)
        Base.metadata.create_all(engine)
        
        click.echo("✅ Database initialized successfully")
        
    except Exception as e:
        click.echo(f"❌ Database initialization failed: {e}")

@database.command()
def migrate():
    """Run database migrations"""
    click.echo("🔄 Running database migrations...")
    
    try:
        import subprocess
        result = subprocess.run(["alembic", "upgrade", "head"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            click.echo("✅ Migrations completed successfully")
        else:
            click.echo(f"❌ Migration failed: {result.stderr}")
            
    except FileNotFoundError:
        click.echo("❌ Alembic not found. Install with: pip install alembic")
    except Exception as e:
        click.echo(f"❌ Migration failed: {e}")

@database.command()
def backup():
    """Create database backup"""
    click.echo("💾 Creating database backup...")
    
    upgrade_mgr = get_upgrade_manager()
    success = upgrade_mgr.backup_database()
    
    if success:
        click.echo("✅ Database backup created successfully")
    else:
        click.echo("❌ Database backup failed")

@cli.group()
def server():
    """Server management commands"""
    pass

@server.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8501, help='Port to bind to')
@click.option('--dev', is_flag=True, help='Run in development mode')
def start(host: str, port: int, dev: bool):
    """Start the CrowdGuard Pro server"""
    click.echo(f"🚀 Starting CrowdGuard Pro server on {host}:{port}")
    
    if dev:
        click.echo("Development mode enabled")
    
    try:
        import subprocess
        cmd = [
            "streamlit", "run", "app.py",
            "--server.address", host,
            "--server.port", str(port),
            "--server.headless", "true" if not dev else "false"
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        click.echo("\n🛑 Server stopped")
    except Exception as e:
        click.echo(f"❌ Failed to start server: {e}")

@server.command()
def status():
    """Check server status"""
    import requests
    
    try:
        settings = get_settings()
        url = f"http://{settings.host}:{settings.port}/_stcore/health"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            click.echo("✅ Server is running")
        else:
            click.echo(f"⚠️  Server responded with status {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        click.echo("❌ Server is not running")
    except Exception as e:
        click.echo(f"❌ Failed to check server status: {e}")

@cli.group()
def backup():
    """Backup management commands"""
    pass

@backup.command()
@click.option('--name', help='Backup name (auto-generated if not provided)')
def create(name: str):
    """Create application backup"""
    upgrade_mgr = get_upgrade_manager()
    
    click.echo("💾 Creating application backup...")
    backup_path = upgrade_mgr.create_backup(name)
    click.echo(f"✅ Backup created: {backup_path}")

@backup.command()
def list():
    """List available backups"""
    upgrade_mgr = get_upgrade_manager()
    backup_dir = upgrade_mgr.backup_dir
    
    if not backup_dir.exists():
        click.echo("No backups found")
        return
    
    backups = [d for d in backup_dir.iterdir() if d.is_dir()]
    
    if not backups:
        click.echo("No backups found")
        return
    
    click.echo("📦 Available backups:")
    for backup in sorted(backups):
        metadata_file = backup / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                click.echo(f"  {backup.name} - {metadata.get('timestamp', 'Unknown')} (v{metadata.get('version', 'Unknown')})")
            except:
                click.echo(f"  {backup.name}")
        else:
            click.echo(f"  {backup.name}")

@backup.command()
@click.argument('backup_name')
def restore(backup_name: str):
    """Restore from backup"""
    upgrade_mgr = get_upgrade_manager()
    
    if not click.confirm(f"This will restore from backup '{backup_name}'. Continue?"):
        click.echo("Restore cancelled")
        return
    
    click.echo(f"🔄 Restoring from backup: {backup_name}")
    success = upgrade_mgr.restore_backup(backup_name)
    
    if success:
        click.echo("✅ Restore completed successfully")
    else:
        click.echo("❌ Restore failed")

@cli.command()
def doctor():
    """Run system diagnostics"""
    click.echo("🏥 Running CrowdGuard Pro diagnostics...")
    click.echo()
    
    issues = []
    
    # Check Python version
    import sys
    python_version = sys.version_info
    if python_version >= (3, 11):
        click.echo("✅ Python version OK")
    else:
        click.echo("❌ Python 3.11+ required")
        issues.append("Python version")
    
    # Check dependencies
    try:
        import streamlit
        import torch
        import cv2
        import numpy as np
        click.echo("✅ Core dependencies OK")
    except ImportError as e:
        click.echo(f"❌ Missing dependency: {e}")
        issues.append("Dependencies")
    
    # Check configuration
    try:
        settings = get_settings()
        click.echo("✅ Configuration loaded OK")
    except Exception as e:
        click.echo(f"❌ Configuration error: {e}")
        issues.append("Configuration")
    
    # Check directories
    required_dirs = ["uploads", "temp", "logs", "models"]
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            click.echo(f"✅ Directory '{dir_name}' exists")
        else:
            click.echo(f"⚠️  Directory '{dir_name}' missing")
    
    click.echo()
    if issues:
        click.echo("⚠️  Issues found:")
        for issue in issues:
            click.echo(f"  • {issue}")
        click.echo("\nRun 'python crowdguard.py --help' for management commands")
    else:
        click.echo("🎉 All checks passed!")

if __name__ == '__main__':
    cli()
