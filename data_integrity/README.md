# DREAMS Data Integrity Layer (Phase-1)

A lightweight, optional validation utility for multimodal time-series data in DREAMS.

## Purpose

This module validates data **before analysis** to catch common issues:
- Malformed JSON structure
- Missing media files
- Future timestamps
- Out-of-order events

**Important**: This layer **only reports issues** — it does NOT modify or "fix" data.

## Quick Start

### Basic Usage

```bash
# From DREAMS root directory
python -m data_integrity.validator \
    --input data/person-01/data.json \
    --base-dir data/
```

### With Schema Validation

```bash
python -m data_integrity.validator \
    --input data/person-01/data.json \
    --schema schemas/sample_schema.json \
    --base-dir data/
```

### Strict Temporal Ordering

```bash
python -m data_integrity.validator \
    --input data/person-01/data.json \
    --base-dir data/ \
    --strict-temporal
```

## Try the Examples

```bash
# Run on example data (will show intentional errors)
python -m data_integrity.validator \
    --input data_integrity/examples/sample_data.json \
    --schema data_integrity/examples/sample_schema.json \
    --base-dir .
```

The example data includes intentional errors to demonstrate validation:
- Out-of-order timestamps
- Future timestamps
- Missing media files

## Architecture

```
data_integrity/
├── __init__.py              # Package initialization
├── validator.py             # CLI entry point and orchestration
├── schema_validator.py      # JSON Schema validation
├── path_validator.py        # Media file existence checks
├── temporal_validator.py    # Timestamp ordering validation
├── reporter.py              # Unified error formatting
└── examples/
    ├── sample_data.json     # Example DREAMS data
    └── sample_schema.json   # Example JSON schema
```

### Modular Design

Each validator is independent and returns a list of `ValidationIssue` objects. This makes it easy to:
- Add new validators (e.g., geo-validators, embedding validators)
- Run validators selectively
- Customize error handling

## Validation Checks

### Schema Validation
- Validates JSON structure against optional schema
- Falls back gracefully if `jsonschema` is not installed
- Schema is **optional** — not enforced

### Path Validation
- Checks that all referenced media files exist
- Supports common field names: `image`, `audio`, `video`, `media`, etc.
- Uses pathlib for cross-platform compatibility
- Resolves paths relative to `--base-dir`

### Temporal Validation
- Detects future timestamps
- Detects out-of-order events in sequences
- Optional strict mode (no duplicate timestamps)
- Handles ISO 8601 strings and Unix timestamps

## Exit Codes

- `0` - All validations passed (or only warnings)
- `1` - Validation errors found

## Options

```
--input, -i       Path to input JSON data file (required)
--schema, -s      Path to JSON schema file (optional)
--base-dir, -b    Base directory for media paths (default: current directory)
--strict-temporal Require strictly increasing timestamps
--quiet, -q       Only show errors (suppress warnings/info)
```

## Design Philosophy

### Phase-1 Foundation
This is intentionally minimal and focused. Future phases may add:
- Geo-location validation
- Embedding quality checks
- Multi-person proximity analysis
- Statistical anomaly detection

### Non-Invasive
- **Optional**: Contributors can ignore this layer
- **Read-only**: Never modifies data
- **Informative**: Clear error messages with location context

### Extensible
Adding a new validator:
1. Create a module (e.g., `geo_validator.py`)
2. Implement a function returning `List[ValidationIssue]`
3. Call it from `validator.py`

## Dependencies

**Required:**
- Python 3.7+
- Standard library only (pathlib, json, datetime)

**Optional:**
- `jsonschema` - for schema validation (graceful fallback if missing)

## Testing

The example data includes intentional issues for testing:

```bash
# Should report 4 errors:
# - Out-of-order timestamp (s04 < s03)
# - Future timestamp (s05)
# - Missing audio file (s04)
# - Missing image file (s05)

python -m data_integrity.validator \
    --input data_integrity/examples/sample_data.json \
    --base-dir .
```

## Integration Examples

### Pre-Analysis Check

```python
from data_integrity.validator import main
import sys

# Run validation before analysis
result = main()
if result != 0:
    print("Data validation failed. Please fix issues before analysis.")
    sys.exit(1)

# Proceed with analysis...
```

### Programmatic Use

```python
import json
from pathlib import Path
from data_integrity.reporter import ValidationReport
from data_integrity.schema_validator import validate_schema
from data_integrity.path_validator import validate_paths
from data_integrity.temporal_validator import validate_temporal

# Load data
with open("data.json") as f:
    data = json.load(f)

# Run validators
report = ValidationReport()
report.extend(validate_schema(data, Path("schema.json")))
report.extend(validate_paths(data, Path(".")))
report.extend(validate_temporal(data))

# Check results
if report.has_errors():
    print(report.format_summary())
```

## Contributing

When extending this layer:
- Keep validators independent
- Return `List[ValidationIssue]`
- Use clear, actionable error messages
- Include location context
- Avoid domain-specific assumptions
- Document the validation logic

## License

Same as DREAMS project.
