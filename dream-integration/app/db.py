from pymongo import MongoClient
from gridfs import GridFS
import os
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))

db = client["dreams"]

users_col = db.users
samples_col = db.samples
results_col = db.results
fs = GridFS(db)
