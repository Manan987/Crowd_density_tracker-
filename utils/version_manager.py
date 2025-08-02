"""
Version management for CrowdGuard Pro
Handles version information, compatibility checks, and upgrade paths
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import pkg_resources

@dataclass
class Version:
    """Version information structure"""
    major: int
    minor: int
    patch: int
    pre_release: Optional[str] = None
    build: Optional[str] = None
    
    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.pre_release:
            version += f"-{self.pre_release}"
        if self.build:
            version += f"+{self.build}"
        return version
    
    def __lt__(self, other: 'Version') -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
    
    def __eq__(self, other: 'Version') -> bool:
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)
    
    @classmethod
    def from_string(cls, version_str: str) -> 'Version':
        """Parse version string into Version object"""
        # Handle pre-release and build metadata
        if '+' in version_str:
            version_str, build = version_str.split('+', 1)
        else:
            build = None
            
        if '-' in version_str:
            version_str, pre_release = version_str.split('-', 1)
        else:
            pre_release = None
            
        parts = version_str.split('.')
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {version_str}")
            
        return cls(
            major=int(parts[0]),
            minor=int(parts[1]),
            patch=int(parts[2]),
            pre_release=pre_release,
            build=build
        )

class VersionManager:
    """Manages application versioning and compatibility"""
    
    def __init__(self, version_file: str = "version.json"):
        self.version_file = Path(version_file)
        self.current_version = self._load_current_version()
        self.compatibility_matrix = self._load_compatibility_matrix()
    
    def _load_current_version(self) -> Version:
        """Load current version from file or package"""
        try:
            # Try to load from version.json first
            if self.version_file.exists():
                with open(self.version_file, 'r') as f:
                    data = json.load(f)
                    return Version.from_string(data['version'])
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            pass
        
        try:
            # Fallback to package version
            return Version.from_string(pkg_resources.get_distribution("crowd-density-tracker").version)
        except:
            # Default version if nothing else works
            return Version(1, 0, 0)
    
    def _load_compatibility_matrix(self) -> Dict:
        """Load version compatibility matrix"""
        default_matrix = {
            "model_versions": {
                "1.0.0": ["1.0.0", "1.0.1", "1.0.2"],
                "1.1.0": ["1.1.0", "1.1.1"],
            },
            "database_versions": {
                "1.0.0": ["1.0.0", "1.1.0"],
                "1.1.0": ["1.1.0"],
            },
            "api_versions": {
                "v1": ["1.0.0", "1.1.0"],
                "v2": ["1.1.0"],
            }
        }
        
        try:
            with open("compatibility.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return default_matrix
    
    def save_version_info(self, additional_info: Optional[Dict] = None) -> None:
        """Save current version information to file"""
        version_info = {
            "version": str(self.current_version),
            "release_date": datetime.now().isoformat(),
            "python_version": os.sys.version,
            "dependencies": self._get_dependency_versions(),
        }
        
        if additional_info:
            version_info.update(additional_info)
        
        with open(self.version_file, 'w') as f:
            json.dump(version_info, f, indent=2)
    
    def _get_dependency_versions(self) -> Dict[str, str]:
        """Get versions of key dependencies"""
        key_dependencies = [
            "streamlit", "torch", "opencv-python", "numpy", 
            "pandas", "plotly", "sqlalchemy"
        ]
        
        versions = {}
        for dep in key_dependencies:
            try:
                versions[dep] = pkg_resources.get_distribution(dep).version
            except:
                versions[dep] = "unknown"
        
        return versions
    
    def check_compatibility(self, component: str, target_version: str) -> bool:
        """Check if current version is compatible with target component version"""
        if component not in self.compatibility_matrix:
            return True  # Assume compatible if no matrix defined
        
        current_str = str(self.current_version)
        compatible_versions = self.compatibility_matrix[component].get(target_version, [])
        
        return current_str in compatible_versions
    
    def get_upgrade_path(self, target_version: str) -> List[str]:
        """Get recommended upgrade path to target version"""
        target = Version.from_string(target_version)
        current = self.current_version
        
        if current >= target:
            return []
        
        # Simple upgrade path - could be more sophisticated
        upgrade_steps = []
        
        # Major version upgrades
        for major in range(current.major, target.major + 1):
            if major > current.major:
                upgrade_steps.append(f"{major}.0.0")
        
        # Minor version upgrades
        if target.major == current.major:
            for minor in range(current.minor + 1, target.minor + 1):
                upgrade_steps.append(f"{target.major}.{minor}.0")
        
        # Final target version
        if str(target) not in upgrade_steps:
            upgrade_steps.append(str(target))
        
        return upgrade_steps
    
    def validate_upgrade(self, target_version: str) -> Tuple[bool, List[str]]:
        """Validate if upgrade to target version is possible"""
        issues = []
        
        try:
            target = Version.from_string(target_version)
        except ValueError as e:
            return False, [f"Invalid target version format: {e}"]
        
        if target < self.current_version:
            issues.append("Target version is older than current version")
        
        # Check dependency compatibility
        for component in ["model_versions", "database_versions"]:
            if not self.check_compatibility(component, target_version):
                issues.append(f"Incompatible {component} for target version")
        
        return len(issues) == 0, issues
    
    def get_changelog(self, from_version: Optional[str] = None) -> str:
        """Get changelog between versions"""
        # This would typically read from a CHANGELOG.md file
        changelog_file = Path("CHANGELOG.md")
        if changelog_file.exists():
            return changelog_file.read_text()
        
        return "No changelog available"

# Global version manager instance
version_manager = VersionManager()

def get_version() -> str:
    """Get current application version"""
    return str(version_manager.current_version)

def get_version_info() -> Dict:
    """Get detailed version information"""
    return {
        "version": str(version_manager.current_version),
        "dependencies": version_manager._get_dependency_versions(),
        "python_version": os.sys.version,
    }
