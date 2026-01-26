from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime, timezone
import logging

def extract_gps_from_image(image_path):
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
                if not (isinstance(val, (tuple, list)) and len(val) == 3):
                    raise ValueError("GPS coordinate value is not a valid 3-element tuple.")
                d, m, s = val
                return float(d) + float(m)/60.0 + float(s)/3600.0
            
            lat = to_degrees(gps_info["GPSLatitude"])
            if gps_info.get("GPSLatitudeRef") == "S":
                lat = -lat
                
            lon = to_degrees(gps_info["GPSLongitude"])
            if gps_info.get("GPSLongitudeRef") == "W":
                lon = -lon
            
            result = {"lat": lat, "lon": lon}
            
            timestamp = None
            if datetime_original:
                try:
                    # EXIF DateTimeOriginal has no timezone, parse as naive and convert to ISO format
                    timestamp = datetime.strptime(datetime_original, '%Y:%m:%d %H:%M:%S').isoformat()
                except (ValueError, TypeError):
                    logging.warning(f"Could not parse EXIF DateTimeOriginal: '{datetime_original}'")
            
            if not timestamp and 'GPSDateStamp' in gps_info and 'GPSTimeStamp' in gps_info:
                try:
                    date_str = gps_info['GPSDateStamp']
                    time_parts = gps_info['GPSTimeStamp']
                    h, m, s = [float(x) for x in time_parts]
                    # GPS time is specified in UTC
                    ts_str = f"{date_str} {int(h):02}:{int(m):02}:{int(s):02}"
                    dt_utc = datetime.strptime(ts_str, '%Y:%m:%d %H:%M:%S').replace(tzinfo=timezone.utc)
                    timestamp = dt_utc.isoformat()
                except (ValueError, TypeError, IndexError):
                    logging.warning("Could not parse GPSDateStamp and GPSTimeStamp.")
            
            if timestamp:
                result["timestamp"] = timestamp
            return result
        
    except (AttributeError, KeyError, IndexError, TypeError, ValueError, IOError) as e:
        logging.error(f"Failed to extract GPS from '{image_path}': {e}")
        return None
