# EXIF Extraction Research

## Overview

**Note**: EXIF extraction has been implemented in `dreamsApp/exif_extractor.py` by PR #77 (kunal-595). This research document provided the foundation for that implementation.

This document compares EXIF extraction libraries for photo metadata analysis in the DREAMS project, focusing on location data, timestamps, and camera information needed for recovery journey tracking.

## Library Comparison: Pillow vs exifread

### Pillow (PIL.ExifTags)

**Pros:**
- Built into PIL/Pillow (already used for image processing)
- Simple API with `Image._getexif()`
- Good for basic EXIF data
- Lightweight for standard use cases

**Cons:**
- Limited EXIF tag support
- No GPS coordinate parsing helpers
- Inconsistent handling of malformed data
- Returns numeric tag IDs requiring manual mapping

**Code Example:**
```python
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def extract_exif_pillow(image_path):
    image = Image.open(image_path)
    exif = image._getexif()
    if not exif:
        return {}
    
    data = {}
    for tag_id, value in exif.items():
        tag = TAGS.get(tag_id, tag_id)
        if tag == 'GPSInfo':
            gps_data = {}
            for gps_tag_id, gps_value in value.items():
                gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                gps_data[gps_tag] = gps_value
            data[tag] = gps_data
        else:
            data[tag] = value
    return data
```

### exifread

**Pros:**
- Comprehensive EXIF tag support
- Better handling of malformed/corrupted data
- Detailed GPS parsing
- More robust for edge cases
- Returns human-readable tag names

**Cons:**
- Additional dependency
- Slightly more complex API
- Larger memory footprint

**Code Example:**
```python
import exifread

def extract_exif_exifread(image_path):
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)
    
    data = {}
    for tag, value in tags.items():
        if tag.startswith('GPS'):
            data[tag] = str(value)
        elif tag in ['EXIF DateTime', 'Image DateTime']:
            data[tag] = str(value)
        elif tag == 'Image Make':
            data[tag] = str(value)
    return data
```

## Edge Cases Identified

### 1. Missing GPS Data
- **Issue:** Many photos lack GPS coordinates
- **Impact:** Cannot determine location for proximity analysis
- **Mitigation:** Fallback to user-provided location or skip location-based features

### 2. Corrupted EXIF Headers
- **Issue:** Malformed EXIF data causes parsing failures
- **Impact:** Complete metadata loss
- **Mitigation:** Use exifread's robust parsing + try/catch blocks

### 3. Timezone Inconsistencies
- **Issue:** EXIF timestamps don't include timezone info
- **Impact:** Incorrect temporal ordering across locations
- **Mitigation:** Use GPS coordinates to infer timezone or prompt user

### 4. Camera-Specific Formats
- **Issue:** Different manufacturers use proprietary EXIF extensions
- **Impact:** Inconsistent metadata availability
- **Mitigation:** Normalize to common subset of tags

### 5. Privacy-Stripped Images
- **Issue:** Social media platforms remove EXIF data
- **Impact:** No metadata available for analysis
- **Mitigation:** Detect stripped images and request manual input

### 6. Large File Handling
- **Issue:** High-resolution images may cause memory issues
- **Impact:** Processing failures on resource-constrained systems
- **Mitigation:** Stream processing or thumbnail extraction

## Recommended Implementation

### Primary Choice: exifread
- Better edge case handling
- More comprehensive GPS support
- Robust parsing for corrupted data

### Fallback Strategy
```python
def extract_metadata(image_path):
    try:
        return extract_exif_exifread(image_path)
    except Exception:
        try:
            return extract_exif_pillow(image_path)
        except Exception:
            return {}  # No metadata available
```

## GPS Coordinate Conversion

Both libraries require manual GPS coordinate conversion:

```python
def convert_gps_to_decimal(gps_coord, direction):
    """Convert GPS coordinates from DMS to decimal degrees."""
    if not gps_coord:
        return None
    
    degrees = float(gps_coord[0])
    minutes = float(gps_coord[1])
    seconds = float(gps_coord[2])
    
    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    
    if direction in ['S', 'W']:
        decimal = -decimal
    
    return decimal
```

## Testing Strategy

### Test Cases Required
1. **Standard photos** with complete EXIF
2. **GPS-enabled photos** from different devices
3. **Corrupted EXIF** data scenarios
4. **Privacy-stripped** images
5. **Various camera manufacturers** (Canon, Nikon, iPhone, Android)
6. **Different file formats** (JPEG, TIFF, RAW)

### Performance Benchmarks
- Processing time per image
- Memory usage with large files
- Error handling robustness

## Integration with DREAMS

### Metadata Schema
```python
{
    "timestamp": "2024-01-15T14:30:00",
    "location": {
        "lat": 61.2181,
        "lon": -149.9003,
        "accuracy": "high"  # high/medium/low/none
    },
    "camera": {
        "make": "Apple",
        "model": "iPhone 12",
        "settings": {...}
    },
    "processing": {
        "exif_source": "exifread",  # exifread/pillow/manual
        "extraction_time": "2024-01-15T14:35:00"
    }
}
```

### Error Handling
- Log extraction failures for debugging
- Graceful degradation when metadata unavailable
- User prompts for critical missing data (location, timestamp)

## Next Steps

1. Implement robust EXIF extraction module
2. Create comprehensive test suite
3. Add GPS coordinate validation
4. Integrate with photo upload pipeline
5. Add user interface for manual metadata entry