"""
Media path validation for DREAMS data.

Verifies that referenced image, audio, and video files exist on the filesystem.
Uses pathlib for cross-platform compatibility.
"""

from typing import List, Dict, Any
from pathlib import Path

from .reporter import ValidationIssue, Severity


def validate_paths(data: dict, base_dir: Path) -> List[ValidationIssue]:
    """
    Validate that all media file paths exist.
    
    Args:
        data: The data containing media paths
        base_dir: Base directory for resolving relative paths
    
    Returns:
        List of validation issues
    """
    issues = []
    
    # Extract paths from data structure
    paths_to_check = _extract_media_paths(data)
    
    for path_info in paths_to_check:
        path_str = path_info["path"]
        location = path_info["location"]
        
        if not path_str:
            issues.append(ValidationIssue(
                severity=Severity.WARNING,
                category="path",
                message="Empty media path",
                location=location
            ))
            continue
        
        # Skip remote URLs (for CI/CD and cloud storage)
        if _is_remote_url(path_str):
            issues.append(ValidationIssue(
                severity=Severity.INFO,
                category="path",
                message="Skipping remote URL validation",
                location=location,
                details={"url": path_str}
            ))
            continue
        
        # Resolve path relative to base_dir
        path = Path(path_str)
        if not path.is_absolute():
            path = base_dir / path
        
        # Check existence
        if not path.exists():
            issues.append(ValidationIssue(
                severity=Severity.ERROR,
                category="path",
                message=f"Media file not found: {path_str}",
                location=location,
                details={"resolved_path": str(path)}
            ))
        elif not path.is_file():
            issues.append(ValidationIssue(
                severity=Severity.ERROR,
                category="path",
                message=f"Path exists but is not a file: {path_str}",
                location=location,
                details={"resolved_path": str(path)}
            ))
    
    return issues


def _is_remote_url(path_str: str) -> bool:
    """
    Check if a path is a remote URL.
    
    Args:
        path_str: Path string to check
    
    Returns:
        True if path is a remote URL, False otherwise
    """
    remote_schemes = ("http://", "https://", "s3://", "ftp://")
    return path_str.lower().startswith(remote_schemes)


def _extract_media_paths(data: dict, parent_key: str = "") -> List[Dict[str, str]]:
    """
    Recursively extract media paths from nested data structure.
    
    Looks for common media field names: image, audio, video, media, file_path, etc.
    
    Args:
        data: Data structure to search
        parent_key: Parent location for building path strings
    
    Returns:
        List of dicts with 'path' and 'location' keys
    """
    media_fields = {
        "image", "audio", "video", "media", 
        "image_path", "audio_path", "video_path", "file_path",
        "img", "sound", "recording"
    }
    
    paths = []
    
    if isinstance(data, dict):
        for key, value in data.items():
            location = f"{parent_key}.{key}" if parent_key else key
            
            # Check if this key is a media field
            if key.lower() in media_fields and isinstance(value, str):
                paths.append({"path": value, "location": location})
            
            # Recurse into nested structures
            elif isinstance(value, (dict, list)):
                paths.extend(_extract_media_paths(value, location))
    
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            location = f"{parent_key}[{idx}]"
            paths.extend(_extract_media_paths(item, location))
    
    return paths
