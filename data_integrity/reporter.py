"""
Unified error reporting for validation issues.

All validators return lists of ValidationIssue objects for consistent reporting.
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class Severity(Enum):
    """Issue severity levels."""
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    severity: Severity
    category: str  # e.g., "schema", "path", "temporal"
    message: str
    location: Optional[str] = None  # e.g., "samples[3]", "person_id: p01"
    details: Optional[dict] = None
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "severity": self.severity.value,
            "category": self.category,
            "message": self.message,
            "location": self.location,
            "details": self.details
        }


class ValidationReport:
    """Aggregates and formats validation issues."""
    
    def __init__(self):
        self.issues: List[ValidationIssue] = []
    
    def add(self, issue: ValidationIssue):
        """Add a validation issue."""
        self.issues.append(issue)
    
    def extend(self, issues: List[ValidationIssue]):
        """Add multiple validation issues."""
        self.issues.extend(issues)
    
    def has_errors(self) -> bool:
        """Check if any errors exist."""
        return any(issue.severity == Severity.ERROR for issue in self.issues)
    
    def count_by_severity(self) -> dict:
        """Count issues by severity."""
        counts = {severity: 0 for severity in Severity}
        for issue in self.issues:
            counts[issue.severity] += 1
        return counts
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary for CI/CD pipelines."""
        counts = self.count_by_severity()
        return {
            "summary": {
                "total_issues": len(self.issues),
                "errors": counts[Severity.ERROR],
                "warnings": counts[Severity.WARNING],
                "info": counts[Severity.INFO],
                "has_errors": self.has_errors()
            },
            "issues": [issue.to_dict() for issue in self.issues]
        }
    
    def format_summary(self) -> str:
        """Generate human-readable summary."""
        if not self.issues:
            return "All validation checks passed."
        
        counts = self.count_by_severity()
        lines = [
            "\n" + "=" * 60,
            "VALIDATION REPORT",
            "=" * 60,
        ]
        
        for issue in self.issues:
            icon = "X" if issue.severity == Severity.ERROR else "!" if issue.severity == Severity.WARNING else "i"
            location_str = f" [{issue.location}]" if issue.location else ""
            lines.append(f"\n{icon} {issue.severity.value} ({issue.category}){location_str}")
            lines.append(f"  {issue.message}")
            if issue.details:
                for key, value in issue.details.items():
                    lines.append(f"    {key}: {value}")
        
        lines.append("\n" + "-" * 60)
        lines.append(f"Summary: {counts[Severity.ERROR]} errors, "
                    f"{counts[Severity.WARNING]} warnings, "
                    f"{counts[Severity.INFO]} info")
        lines.append("=" * 60 + "\n")
        
        return "\n".join(lines)
