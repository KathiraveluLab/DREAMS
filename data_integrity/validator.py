"""
DREAMS Data Integrity Validator - CLI Entry Point

Usage:
    python -m data_integrity.validator --input data.json [options]
    
Example:
    python -m data_integrity.validator \\
        --input examples/sample_data.json \\
        --schema examples/sample_schema.json \\
        --base-dir ./
"""

import argparse
import json
import logging
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .reporter import ValidationReport, Severity
from .schema_validator import validate_schema
from .path_validator import validate_paths
from .temporal_validator import validate_temporal


logger = logging.getLogger(__name__)


# Dependency validation — extends Data Integrity Layer from PR #38
def validate_dependencies() -> dict:
    """Check required distributions from requirements.txt and report missing ones.

    This function reads requirements from the project root, extracts package
    names while ignoring version specifiers/comments, and verifies installation
    using importlib.metadata.version(). It never raises for missing packages;
    instead it logs warnings and returns a status dictionary.
    """
   
    requirements_path = Path(__file__).resolve().parent.parent / "requirements.txt"
    if not requirements_path.exists():
        logger.warning(
            "WARNING: requirements.txt not found at %s. Skipping dependency validation.",
            requirements_path,
        )
        return {"valid": True, "missing": []}

    missing_packages = []

    with requirements_path.open("r", encoding="utf-8") as requirements_file:
        for raw_line in requirements_file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            # Remove inline comments and environment markers.
            line = line.split("#", 1)[0].strip()
            line = line.split(";", 1)[0].strip()
            if not line:
                continue

            # Ignore pip options and nested requirements directives.
            if line.startswith("-"):
                continue

            package_name = line
            for separator in ("@", "==", ">=", "<=", "~=", "!=", ">", "<"):
                if separator in package_name:
                    package_name = package_name.split(separator, 1)[0].strip()
                    break

            # Ignore extras in requirement specifiers (e.g., package[extra]).
            package_name = package_name.split("[", 1)[0].strip()
            if not package_name:
                continue

            try:
                version(package_name)
            except PackageNotFoundError:
                missing_packages.append(package_name)
                logger.warning(
                    "WARNING: Required package %s is not installed. Run pip install -r requirements.txt",
                    package_name,
                )

    return {"valid": len(missing_packages) == 0, "missing": missing_packages}


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="DREAMS Data Integrity Validator (Phase-1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate with schema
  python -m data_integrity.validator --input data.json --schema schema.json
  
  # Validate without schema
  python -m data_integrity.validator --input data.json --base-dir ./data
  
  # Strict temporal ordering
  python -m data_integrity.validator --input data.json --strict-temporal
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=Path,
        help="Path to input JSON data file"
    )
    
    parser.add_argument(
        "--schema", "-s",
        type=Path,
        help="Path to JSON schema file (optional)"
    )
    
    parser.add_argument(
        "--base-dir", "-b",
        type=Path,
        default=Path("."),
        help="Base directory for resolving relative media paths (default: current directory)"
    )
    
    parser.add_argument(
        "--strict-temporal",
        action="store_true",
        help="Require strictly increasing timestamps (no duplicates)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only print errors, suppress warnings and info"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (for CI/CD pipelines)"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1
    
    # Load data
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}", file=sys.stderr)
        return 1
    
    # Initialize report
    report = ValidationReport()
    
    # Run validators
    print(f"Validating: {args.input}")
    print(f"Base directory: {args.base_dir.resolve()}")
    if args.schema:
        print(f"Schema: {args.schema}")
    print()
    
    # Schema validation
    if args.schema:
        print("Running schema validation...")
        issues = validate_schema(data, args.schema)
        report.extend(issues)
    
    # Path validation
    print("Running path validation...")
    issues = validate_paths(data, args.base_dir)
    report.extend(issues)
    
    # Temporal validation
    print("Running temporal validation...")
    issues = validate_temporal(data, strict_monotonic=args.strict_temporal)
    report.extend(issues)
    
    # Filter by severity if quiet mode
    if args.quiet:
        report.issues = [i for i in report.issues if i.severity == Severity.ERROR]
    
    # Print report in requested format
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.format_summary())
    
    # Exit with appropriate code
    return 1 if report.has_errors() else 0


if __name__ == "__main__":
    sys.exit(main())
