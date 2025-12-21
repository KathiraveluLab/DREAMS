import os
import json
from pymongo import MongoClient
from gridfs import GridFS



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    raise RuntimeError("MONGO_URI environment variable not set. Please create a .env file with it.")
client = MongoClient(mongo_uri)

db = client["dreams"]

users_col = db.users
samples_col = db.samples
results_col = db.results
fs = GridFS(db)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
AUDIO_EXTS = {".mp3", ".wav"}



def store_file_gridfs(path):
    with open(path, "rb") as f:
        return fs.put(f, filename=os.path.basename(path))

def read_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()



def migrate():
    if not os.path.isdir(DATA_DIR):
        print("❌ data/ directory not found")
        return

    for person in os.listdir(DATA_DIR):
        person_dir = os.path.join(DATA_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        print(f"\n👤 Migrating person: {person}")

        users_col.update_one(
            {"person_id": person},
            {"$set": {"person_id": person}},
            upsert=True
        )

        for sample in os.listdir(person_dir):
            if not sample.startswith("sample"):
                continue

            sample_dir = os.path.join(person_dir, sample)
            if not os.path.isdir(sample_dir):
                continue

            print(f"  📦 Sample: {sample}")

            image_id = None
            audio_id = None
            transcript = None
            transcript_fallback = None
            description = ""

           
            for file in os.listdir(sample_dir):
                path = os.path.join(sample_dir, file)
                name, ext = os.path.splitext(file)
                ext = ext.lower()

                if ext in IMAGE_EXTS:
                    image_id = store_file_gridfs(path)
                    print(f"    🖼 Image stored: {file}")

                elif ext in AUDIO_EXTS:
                    audio_id = store_file_gridfs(path)
                    print(f"    🔊 Audio stored: {file}")

                elif ext == ".txt":
                    if name.startswith("transcript"):
                        transcript = read_text_file(path)
                        print("    📄 Transcript loaded (transcript*)")

                    elif name.startswith("clip") and transcript_fallback is None:
                        transcript_fallback = read_text_file(path)

                    elif name.startswith("description"):
                        description = read_text_file(path)
                        print("    📝 Description loaded")

           
            if transcript is None and transcript_fallback is not None:
                transcript = transcript_fallback
                print("    📄 Transcript loaded (clip* fallback)")

            samples_col.update_one(
                {"person_id": person, "sample_id": sample},
                {
                    "$set": {
                        "person_id": person,
                        "sample_id": sample,
                        "image_id": image_id,
                        "audio_id": audio_id,
                        "transcript": transcript or "",
                        "description": description
                    }
                },
                upsert=True
            )

           
            analysis_dirs = [
                os.path.join(person_dir, "analysis-p01", sample),
                os.path.join(person_dir, f"analysis-{person}", sample)
            ]

            for a_dir in analysis_dirs:
                if not os.path.isdir(a_dir):
                    continue

                print(f"    📊 Migrating analysis from {a_dir}")

                text_scores = {}
                image_scores = {}

                text_path = os.path.join(a_dir, "text_scores.json")
                image_path = os.path.join(a_dir, "image_scores.json")

                if os.path.isfile(text_path):
                    with open(text_path) as f:
                        text_scores = json.load(f)

                if os.path.isfile(image_path):
                    with open(image_path) as f:
                        image_scores = json.load(f)

                if text_scores or image_scores:
                    results_col.update_one(
                        {"person_id": person, "sample_id": sample},
                        {"$set": {
                            "text_scores": text_scores,
                            "image_scores": image_scores
                        }},
                        upsert=True
                    )

    print("\n✅ Migration completed successfully.")

    print("\n✅ Migration completed successfully.")



if __name__ == "__main__":
    migrate()
