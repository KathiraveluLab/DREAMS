"""EXIF metadata extraction module for photo analysis."""

import exifread
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class EXIFExtractor:
    """Extract and process EXIF metadata from images."""
    
    def extract_metadata(self, image_path):
        """Extract metadata using fallback strategy."""
        try:
            return self._extract_exifread(image_path)
        except Exception as e:
            logger.warning(f"exifread failed: {e}, trying Pillow")
            try:
                return self._extract_pillow(image_path)
            except Exception as e2:
                logger.error(f"Both extractors failed: {e2}")
                return self._empty_metadata()
    
    def _extract_exifread(self, image_path):
        """Extract using exifread library."""
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f)
        
        return {
            "timestamp": self._parse_timestamp(tags),
            "location": self._parse_gps_exifread(tags),
            "camera": self._parse_camera_exifread(tags),
            "processing": {"exif_source": "exifread"}
        }
    
    def _extract_pillow(self, image_path):
        """Extract using Pillow as fallback."""
        image = Image.open(image_path)
        exif = image._getexif()
        
        if not exif:
            return self._empty_metadata()
        
        return {
            "timestamp": self._parse_timestamp_pillow(exif),
            "location": self._parse_gps_pillow(exif),
            "camera": self._parse_camera_pillow(exif),
            "processing": {"exif_source": "pillow"}
        }
    
    def _parse_gps_exifread(self, tags):
        """Parse GPS coordinates from exifread tags."""
        lat = self._get_gps_coordinate(tags, 'GPS GPSLatitude', 'GPS GPSLatitudeRef')
        lon = self._get_gps_coordinate(tags, 'GPS GPSLongitude', 'GPS GPSLongitudeRef')
        
        if lat and lon:
            return {"lat": lat, "lon": lon, "accuracy": "high"}
        return {"accuracy": "none"}
    
    def _get_gps_coordinate(self, tags, coord_key, ref_key):
        """Convert GPS coordinate to decimal degrees."""
        coord = tags.get(coord_key)
        ref = tags.get(ref_key)
        
        if not coord or not ref:
            return None
        
        try:
            # Directly access the 'values' attribute which contains a list of Ratio objects.
            # This is more robust than parsing the string representation.
            if not hasattr(coord, 'values') or len(coord.values) != 3:
                return None

            degrees = float(coord.values[0])
            minutes = float(coord.values[1])
            seconds = float(coord.values[2])

            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)

            if hasattr(ref, 'values') and str(ref.values) in ['S', 'W']:
                decimal = -decimal
            
            return decimal
        except (ValueError, IndexError, TypeError) as e:
            logger.warning(f"Could not parse GPS coordinate: {coord}. Error: {e}")
            return None
    
    def _parse_timestamp(self, tags):
        """Parse timestamp from exifread tags."""
        for key in ['EXIF DateTime', 'Image DateTime', 'EXIF DateTimeOriginal']:
            if key in tags:
                try:
                    dt_str = str(tags[key])
                    return datetime.strptime(dt_str, '%Y:%m:%d %H:%M:%S').isoformat()
                except ValueError:
                    continue
        return None
    
    def _parse_camera_exifread(self, tags):
        """Parse camera info from exifread tags."""
        return {
            "make": str(tags.get('Image Make', '')),
            "model": str(tags.get('Image Model', ''))
        }
    
    def _parse_timestamp_pillow(self, exif):
        """Parse timestamp from Pillow exif data."""
        for tag in [36867, 306, 36868]:  # DateTimeOriginal, DateTime, DateTimeDigitized
            if tag in exif:
                try:
                    dt_str = exif[tag]
                    if isinstance(dt_str, bytes):
                        dt_str = dt_str.decode('utf-8')
                    return datetime.strptime(dt_str, '%Y:%m:%d %H:%M:%S').isoformat()
                except (ValueError, AttributeError):
                    continue
        return None

    def _parse_gps_pillow(self, exif):
        """Parse GPS coordinates from Pillow exif data."""
        if 'GPSInfo' not in exif:
            return {"accuracy": "none"}

        gps_info = exif['GPSInfo']

        def get_coordinate(coord, ref):
            if coord not in gps_info or ref not in gps_info:
                return None
            coord_vals = gps_info[coord]
            ref_val = gps_info[ref]

            try:
                degrees = coord_vals[0][0] / coord_vals[0][1]
                minutes = coord_vals[1][0] / coord_vals[1][1]
                seconds = coord_vals[2][0] / coord_vals[2][1]

                decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)

                if ref_val in [b'S', b'W']:
                    decimal = -decimal

                return decimal
            except (IndexError, TypeError, ZeroDivisionError):
                return None

        lat = get_coordinate(2, 1)  # GPSLatitude, GPSLatitudeRef
        lon = get_coordinate(4, 3)  # GPSLongitude, GPSLongitudeRef

        if lat and lon:
            return {"lat": lat, "lon": lon, "accuracy": "high"}
        return {"accuracy": "none"}

    def _parse_camera_pillow(self, exif):
        """Parse camera info from Pillow exif data."""
        return {
            "make": exif.get(271, "").decode('utf-8') if isinstance(exif.get(271), bytes) else str(exif.get(271, "")),
            "model": exif.get(272, "").decode('utf-8') if isinstance(exif.get(272), bytes) else str(exif.get(272, ""))
        }

    def _empty_metadata(self):
        """Return empty metadata structure."""
        return {
            "timestamp": None,
            "location": {"accuracy": "none"},
            "camera": {"make": "", "model": ""},
            "processing": {"exif_source": "none"}
        }
    
