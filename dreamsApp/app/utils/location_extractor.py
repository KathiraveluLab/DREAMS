from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime, timezone
import logging

def extract_gps_from_image(image_file):
    try:
        with Image.open(image_path) as image:
            info = image.getexif()
            if not info:
                return None
            
            # Parse EXIF for GPS block
            gps_info = None
            datetime_original = None
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                if decoded == "GPSInfo":
                    gps_info = {GPSTAGS.get(t, t): value[t] for t in value}
                elif decoded == "DateTimeOriginal":
                    datetime_original = value
                if gps_info is not None and datetime_original is not None:
                    break
            
            if not gps_info:
                return None
            
            if "GPSLatitude" not in gps_info or "GPSLongitude" not in gps_info:
                return None
            
            def to_degrees(val):
                """Converts GPS coordinates from DMS to decimal degrees."""
                if not isinstance(val, (tuple, list)) or len(val) != 3:
                    raise ValueError(f"Invalid GPS coordinate format: {val}")
                total_degrees = 0.0
                for i, c in enumerate(val):
                    if isinstance(c, tuple):
                        if c[1] == 0:
                            raise ValueError(f"Invalid GPS coordinate component with zero denominator: {c}")
                        total_degrees += (c[0] / c[1]) / (60**i)
                    else:
                        total_degrees += float(c) / (60**i)
                return total_degrees
            
            lat = to_degrees(gps_info["GPSLatitude"])
            if gps_info.get("GPSLatitudeRef") == "S":
                lat = -lat
                
            lon = to_degrees(gps_info["GPSLongitude"])
            if gps_info.get("GPSLongitudeRef") == "W":
                lon = -lon
            
            result = {"lat": lat, "lon": lon}
            
            timestamp = None
            if 'GPSDateStamp' in gps_info and 'GPSTimeStamp' in gps_info:
                try:
                    date_str = gps_info['GPSDateStamp']
                    time_parts = gps_info['GPSTimeStamp']
                    date_str = gps_info['GPSDateStamp']
                    time_parts = gps_info['GPSTimeStamp']
                    year, month, day = map(int, date_str.split(':'))
                    h, m, s_val = [float(part) for part in time_parts]
                    
                    seconds = int(s_val)
                    microseconds = int((s_val - seconds) * 1_000_000)

                    # GPS time is specified in UTC
                    dt_utc = datetime(year, month, day, int(h), int(m), seconds, microseconds, tzinfo=timezone.utc)
                    timestamp = dt_utc.isoformat()
                except (ValueError, TypeError, IndexError):
                    logging.warning("Could not parse GPSDateStamp and GPSTimeStamp.")
            
            if not timestamp and datetime_original:
                try:
                    # EXIF DateTimeOriginal has no timezone, parse as naive and convert to ISO format
                    timestamp = datetime.strptime(datetime_original, '%Y:%m:%d %H:%M:%S').isoformat()
                except (ValueError, TypeError):
                    logging.warning(f"Could not parse EXIF DateTimeOriginal: '{datetime_original}'")
            
            if timestamp:
                result["timestamp"] = timestamp
            return result
        
    except (AttributeError, KeyError, IndexError, TypeError, ValueError, IOError) as e:
        logging.error(f"Failed to extract GPS from '{image_path}': {e}")
        return None
