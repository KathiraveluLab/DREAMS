from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


def get_gps_coordinates(image_path):
    """Extract GPS coordinates from image EXIF data. Returns None if not available."""
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        
        if not exif_data:
            return None
        
        gps_info = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "GPSInfo":
                for gps_tag_id, gps_value in value.items():
                    gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_info[gps_tag] = gps_value
                break
        
        if not gps_info:
            return None
        
        # Convert GPS coordinates to decimal degrees
        lat = gps_info.get("GPSLatitude")
        lat_ref = gps_info.get("GPSLatitudeRef")
        lon = gps_info.get("GPSLongitude")
        lon_ref = gps_info.get("GPSLongitudeRef")
        
        if not all([lat, lat_ref, lon, lon_ref]):
            return None
        
        latitude = float(lat[0]) + float(lat[1]) / 60 + float(lat[2]) / 3600
        longitude = float(lon[0]) + float(lon[1]) / 60 + float(lon[2]) / 3600
        
        if lat_ref == "S":
            latitude = -latitude
        if lon_ref == "W":
            longitude = -longitude
        
        return {"latitude": latitude, "longitude": longitude}
    
    except Exception:
        return None
