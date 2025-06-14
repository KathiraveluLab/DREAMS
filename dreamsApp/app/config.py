import os
from dotenv import load_dotenv

load_dotenv()

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'images')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "dreams")