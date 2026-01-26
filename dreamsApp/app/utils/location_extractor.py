from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

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
                d, m, s = val
                return float(d) + float(m)/60.0 + float(s)/3600.0
            
            lat = to_degrees(gps_info["GPSLatitude"])
            if gps_info.get("GPSLatitudeRef") == "S":
                lat = -lat
                
            lon = to_degrees(gps_info["GPSLongitude"])
            if gps_info.get("GPSLongitudeRef") == "W":
                lon = -lon
            
            result = {"lat": lat, "lon": lon}
            if datetime_original:
                result["timestamp"] = datetime_original
            elif "GPSDateStamp" in gps_info:
                result["timestamp"] = gps_info["GPSDateStamp"]
            return result
        
    except Exception:
        return None
