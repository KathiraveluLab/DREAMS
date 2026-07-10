import os
from dotenv import load_dotenv

load_dotenv()

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'images')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "dreams")

# Security: use environment-provided key outside development.
SECRET_KEY = os.getenv("SECRET_KEY", "dev")

# Limit request body size for uploads (16 MiB default).
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))

# Comma-separated list of allowed image extensions.
ALLOWED_EXTENSIONS = {
    x.strip().lower()
    for x in os.getenv("ALLOWED_EXTENSIONS", "png,jpg,jpeg,gif").split(",")
    if x.strip()
}