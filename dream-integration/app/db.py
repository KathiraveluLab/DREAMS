from pymongo import MongoClient
from gridfs import GridFS
import os
from dotenv import load_dotenv

load_dotenv()

mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    raise RuntimeError("MONGO_URI environment variable not set. Please create a .env file with it.")
client = MongoClient(mongo_uri)

db = client["dreams"]

users_col = db.users
samples_col = db.samples
results_col = db.results
fs = GridFS(db)
