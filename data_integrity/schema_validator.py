"""
JSON Schema validation for DREAMS data structure.

Validates that JSON data conforms to an expected schema.
Falls back gracefully if jsonschema library is not installed.
"""

from typing import List, Optional
from pathlib import Path
import json

from .reporter import ValidationIssue, Severity


def validate_schema(data: dict, schema_path: Optional[Path] = None) -> List[ValidationIssue]:
    """
    Validate data against JSON schema.
    
    Args:
        data: The data to validate
        schema_path: Path to JSON schema file (optional)
    
    Returns:
        List of validation issues
    """
    issues = []
    
    if schema_path is None:
        return issues  # Schema validation is optional
    
    if not schema_path.exists():
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            category="schema",
            message=f"Schema file not found: {schema_path}",
            location="validator"
        ))
        return issues
    
    # Try to load schema
    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)
    except json.JSONDecodeError as e:
        issues.append(ValidationIssue(
            severity=Severity.ERROR,
            category="schema",
            message=f"Invalid JSON in schema file: {e}",
            location=str(schema_path)
        ))
        return issues
    
    # Try to use jsonschema if available
    try:
        import jsonschema
        from jsonschema import Draft7Validator
        
        validator = Draft7Validator(schema)
        errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
        
        for error in errors:
            # Build location path
            location = ".".join(str(p) for p in error.absolute_path) if error.absolute_path else "root"
            
            issues.append(ValidationIssue(
                severity=Severity.ERROR,
                category="schema",
                message=error.message,
                location=location,
                details={"schema_path": str(error.schema_path)}
            ))
    
    except ImportError:
        issues.append(ValidationIssue(
            severity=Severity.INFO,
            category="schema",
            message="jsonschema library not installed - skipping schema validation",
            location="validator",
            details={"hint": "pip install jsonschema"}
        ))
    
    return issues
