"""
Temporal consistency validation for DREAMS data.

Validates timestamp ordering and detects temporal anomalies:
- Future timestamps
- Non-monotonic sequences
- Out-of-order events
"""

import re
from typing import List, Dict, Any, Optional, Tuple
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
    
    # bool is an int subclass, but True/False are flags, not times
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Try Unix timestamp
        try:
            timestamp_value = value
            # Check if timestamp is in milliseconds
            # Year 9999 in seconds = 253402300800
            if timestamp_value > 253402300800:
                timestamp_value = timestamp_value / 1000
            
            return datetime.fromtimestamp(timestamp_value, tz=timezone.utc)
        except (ValueError, OSError, OverflowError):
            # Out-of-range values (1e300, infinity) are not usable timestamps
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
    
    # Compare each field only with the same field of the neighbouring array items
    grouped = _group_into_sequences(timestamps)
    
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


_ARRAY_INDEX = re.compile(r"\[(\d+)\]")


def _sequence_position(location: str) -> Optional[Tuple[str, int]]:
    """
    Split a location into (sequence key, position) using its last array index.

    "samples[2].timestamp" -> ("samples[].timestamp", 2), and
    "sessions[1].readings[0].timestamp" -> ("sessions[1].readings[].timestamp", 0), so the
    items of one array are ordered against each other and nothing else. A location with no
    array index belongs to no sequence and returns None.
    """
    matches = list(_ARRAY_INDEX.finditer(location))
    if not matches:
        return None
    last = matches[-1]
    key = location[:last.start()] + "[]" + location[last.end():]
    return key, int(last.group(1))


def _group_into_sequences(timestamps: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group timestamps into the sequences that must be in order.

    One sequence is one timestamp field across the items of one array, for example
    "samples[0].timestamp", "samples[1].timestamp", ... Different fields of the same record
    (created_at vs capture_time) or of nested arrays are different sequences, because
    nothing says they must be ordered against each other.
    """
    groups: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}

    for ts_info in timestamps:
        position = _sequence_position(ts_info["location"])
        if position is None:
            continue
        key, index = position
        groups.setdefault(key, []).append((index, ts_info))

    return {
        key: [info for _, info in sorted(members, key=lambda member: member[0])]
        for key, members in groups.items()
    }
