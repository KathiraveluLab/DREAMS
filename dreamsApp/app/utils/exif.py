from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


def get_gps_coordinates(image_path):
    """Extract GPS coordinates from image EXIF data. Returns None if not available."""
    try:
        with Image.open(image_path) as img:
            exif_data = img.getexif()
            
            if not exif_data:
                return None
            
            gps_ifd = exif_data.get_ifd(0x8825)
            if not gps_ifd:
                return None
            
            lat = gps_ifd.get(2)
            lat_ref = gps_ifd.get(1)
            lon = gps_ifd.get(4)
            lon_ref = gps_ifd.get(3)
            
            if not all([lat, lat_ref, lon, lon_ref]):
                return None
            
            if len(lat) != 3 or len(lon) != 3:
                return None
            
            def to_decimal(coord):
                return float(coord[0]) + float(coord[1]) / 60 + float(coord[2]) / 3600
            
            latitude = to_decimal(lat)
            longitude = to_decimal(lon)
            
            if lat_ref == "S":
                latitude = -latitude
            if lon_ref == "W":
                longitude = -longitude
            
            return {"latitude": latitude, "longitude": longitude}
    
    except Exception:
        return None
