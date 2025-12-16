"""
Temporal consistency validation for DREAMS data.

Validates timestamp ordering and detects temporal anomalies:
- Future timestamps
- Non-monotonic sequences
- Out-of-order events
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from .reporter import ValidationIssue, Severity


def validate_temporal(data: dict, strict_monotonic: bool = False) -> List[ValidationIssue]:
    """
    Validate temporal consistency of time-ordered data.
    
    Args:
        data: The data containing timestamps
        strict_monotonic: If True, require strictly increasing timestamps (no duplicates)
    
    Returns:
        List of validation issues
    """
    issues = []
    
    # Extract timestamps from data structure
    timestamps = _extract_timestamps(data)
    
    if not timestamps:
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            category="temporal",
            message="No timestamps found in data",
            location="root"
        ))
        return issues
    
    # Check for future timestamps
    now = datetime.now(timezone.utc)
    for ts_info in timestamps:
        ts = ts_info["timestamp"]
        location = ts_info["location"]
        
        if ts > now:
            issues.append(ValidationIssue(
                severity=Severity.ERROR,
                category="temporal",
                message=f"Future timestamp detected: {ts.isoformat()}",
                location=location,
                details={"current_time": now.isoformat()}
            ))
    
    # Check temporal ordering
    issues.extend(_check_ordering(timestamps, strict_monotonic))
    
    return issues


def _extract_timestamps(data: dict, parent_key: str = "") -> List[Dict[str, Any]]:
    """
    Recursively extract timestamps from data structure.
    
    Looks for common timestamp field names and ISO 8601 formatted strings.
    
    Args:
        data: Data structure to search
        parent_key: Parent location for building path strings
    
    Returns:
        List of dicts with 'timestamp', 'location', and 'raw' keys
    """
    timestamp_fields = {
        "timestamp", "time", "datetime", "created_at", "recorded_at",
        "date", "ts", "event_time", "capture_time"
    }
    
    timestamps = []
    
    if isinstance(data, dict):
        for key, value in data.items():
            location = f"{parent_key}.{key}" if parent_key else key
            
            # Check if this is a timestamp field
            if key.lower() in timestamp_fields:
                ts = _parse_timestamp(value)
                if ts:
                    timestamps.append({
                        "timestamp": ts,
                        "location": location,
                        "raw": value
                    })
            
            # Recurse into nested structures
            elif isinstance(value, (dict, list)):
                timestamps.extend(_extract_timestamps(value, location))
    
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            location = f"{parent_key}[{idx}]"
            timestamps.extend(_extract_timestamps(item, location))
    
    return timestamps


def _parse_timestamp(value: Any) -> Optional[datetime]:
    """
    Parse a timestamp from various formats.
    
    Supports:
    - ISO 8601 strings
    - Unix timestamps in seconds (int/float)
    - Unix timestamps in milliseconds (auto-detected)
    - datetime objects
    
    Returns:
        datetime object with timezone, or None if parsing fails
    """
    if isinstance(value, datetime):
        # Ensure timezone awareness
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    
    if isinstance(value, str):
        # Try ISO 8601 format
        try:
            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, AttributeError):
            pass
    
    if isinstance(value, (int, float)):
        # Try Unix timestamp
        try:
            timestamp_value = value
            # Check if timestamp is in milliseconds
            # Year 9999 in seconds = 253402300800
            if timestamp_value > 253402300800:
                timestamp_value = timestamp_value / 1000
            
            return datetime.fromtimestamp(timestamp_value, tz=timezone.utc)
        except (ValueError, OSError):
            pass
    
    return None


def _check_ordering(timestamps: List[Dict[str, Any]], strict: bool) -> List[ValidationIssue]:
    """
    Check if timestamps are in correct temporal order.
    
    Args:
        timestamps: List of timestamp info dicts
        strict: If True, require strictly increasing (no duplicates)
    
    Returns:
        List of validation issues
    """
    issues = []
    
    if len(timestamps) < 2:
        return issues
    
    # Sort by location to check sequence within arrays
    # Group by parent array (e.g., all items in "samples")
    grouped = _group_by_parent_array(timestamps)
    
    for group_name, group_timestamps in grouped.items():
        for i in range(1, len(group_timestamps)):
            prev = group_timestamps[i - 1]
            curr = group_timestamps[i]
            
            prev_ts = prev["timestamp"]
            curr_ts = curr["timestamp"]
            
            if strict:
                if curr_ts <= prev_ts:
                    issues.append(ValidationIssue(
                        severity=Severity.ERROR,
                        category="temporal",
                        message=f"Non-strictly-monotonic timestamps: {curr_ts.isoformat()} <= {prev_ts.isoformat()}",
                        location=f"{curr['location']} (previous: {prev['location']})",
                        details={
                            "current": curr["raw"],
                            "previous": prev["raw"]
                        }
                    ))
            else:
                if curr_ts < prev_ts:
                    issues.append(ValidationIssue(
                        severity=Severity.ERROR,
                        category="temporal",
                        message=f"Out-of-order timestamps: {curr_ts.isoformat()} < {prev_ts.isoformat()}",
                        location=f"{curr['location']} (previous: {prev['location']})",
                        details={
                            "current": curr["raw"],
                            "previous": prev["raw"]
                        }
                    ))
    
    return issues


def _group_by_parent_array(timestamps: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group timestamps by their parent array for sequential checking.
    
    For example, "samples[0].timestamp" and "samples[1].timestamp" 
    should be in the same group.
    """
    groups = {}
    
    for ts_info in timestamps:
        location = ts_info["location"]
        
        # Extract parent array (e.g., "samples" from "samples[0].timestamp")
        if '[' in location:
            parent = location.split('[')[0]
        else:
            parent = "root"
        
        if parent not in groups:
            groups[parent] = []
        groups[parent].append(ts_info)
    
    # Sort each group by index
    for parent, group in groups.items():
        group.sort(key=lambda x: _extract_index(x["location"]))
    
    return groups


def _extract_index(location: str) -> int:
    """Extract array index from location string."""
    if '[' not in location:
        return 0
    
    try:
        start = location.index('[')
        end = location.index(']', start)
        return int(location[start + 1:end])
    except (ValueError, IndexError):
        return 0
