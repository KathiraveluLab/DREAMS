import cmd
import os
import json
import glob
import subprocess
import sys
from flask import Flask, render_template, request, redirect, url_for, flash, Response

from db import users_col, samples_col, results_col, fs

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "dev"


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
ANALYSIS_SCRIPTS_DIR = os.path.join(BASE_DIR, "analysis")



def list_persons():
    return sorted(u["person_id"] for u in users_col.find())

def list_samples(person):
    return sorted(
        s["sample_id"]
        for s in samples_col.find({"person_id": person})
    )

def get_sample(person, sample):
    return samples_col.find_one(
        {"person_id": person, "sample_id": sample}
    )

def read_text(text):
    return text or ""

def read_scores(person, sample, key):
    r = results_col.find_one(
        {"person_id": person, "sample_id": sample}
    )
    return r.get(key, {}) if r else {}


@app.route("/media/image/<person>/<sample>")
def serve_image(person, sample):
    s = get_sample(person, sample)
    if not s or "image_id" not in s:
        return "", 404

    file = fs.get(s["image_id"])
    return Response(file.read(), mimetype="image/png")



@app.route("/", methods=["GET", "POST"])
def index():
    persons = list_persons()
    if not persons:
        return "No persons found in MongoDB.", 200
    

    person = request.values.get("person", persons[0])
    samples = list_samples(person)
    if not samples:
        return f"No samples found for {person}.", 200

    sample = request.values.get("sample", samples[0])
    s = get_sample(person, sample)

    context = {
        "persons": persons,
        "samples": samples,
        "selected_person": person,
        "selected_sample": sample,
        "img_url": url_for("serve_image", person=person, sample=sample)
                   if s and "image_id" in s else None,
        "transcript_text": read_text(s.get("transcript") if s else ""),
        "description_text": read_text(s.get("description") if s else ""),
        "text_scores": read_scores(person, sample, "text_scores"),
        "image_scores": read_scores(person, sample, "image_scores"),
    }

    return render_template("index.html", **context)



@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Analysis scripts located in ./analysis/ with the selected person/sample.
    text_analysis.py:
        python analysis/text_analysis.py --transcript <path> --description <path> --output <json>
        (As text_scores.json)
    image_analysis.py:
        python analysis/image_analysis.py --image <path> --output <json>
    
    Analysis scripts REQUIRE filesystem paths.
    We keep data/ as a staging area.
    """
    person = request.form["person"]
    sample = request.form["sample"]

    sample_dir = os.path.join(DATA_DIR, person, sample)
    person_dir = os.path.join(DATA_DIR, person)
    os.makedirs(sample_dir, exist_ok=True)

    
    image_patterns = ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.webp"]
    
    def _find_first_file(directory, patterns):
        for pattern in patterns:
            matches = glob.glob(os.path.join(directory, pattern))
            if matches:
                return matches[0]
        return None
# In analyze():
    img_path = _find_first_file(sample_dir, ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.webp"])
    transcript_path = _find_first_file(sample_dir, ["transcript*.txt", "clip-*.txt"])
    description_path = _find_first_file(sample_dir, ["description*.txt"])

    transcript_path = None
    for pattern in ("transcript*.txt", "clip-*.txt"):
        matches = glob.glob(os.path.join(sample_dir, pattern))
        if matches:
            transcript_path = matches[0]
            break

    description_matches = glob.glob(os.path.join(sample_dir, "description*.txt"))
    description_path = description_matches[0] if description_matches else None


    out_dir = os.path.join(sample_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    # Text analysis
    text_out = os.path.join(out_dir, "text_scores.json")
    if not os.path.exists(text_out) and (transcript_path or description_path):
        cmd = [
            sys.executable,
            os.path.join(ANALYSIS_SCRIPTS_DIR, "text_analysis.py"),
            "--output", text_out
        ]
        if transcript_path:
            cmd += ["--transcript", transcript_path]
        if description_path:
            cmd += ["--description", description_path]

        try:
            subprocess.run(cmd, check=True)

    
            legacy_candidate = os.path.join(
                person_dir, "analysis-p01", sample, "text_scores.json"
            )
            if not os.path.exists(text_out) and os.path.exists(legacy_candidate):
                os.makedirs(os.path.dirname(text_out), exist_ok=True)
                os.replace(legacy_candidate, text_out)

        except subprocess.CalledProcessError as e:
            flash(f"Text analysis failed: {e}", "error")



    # Image analysis
    img_out = os.path.join(out_dir, "image_scores.json")
    if img_path and not os.path.exists(img_out):
        cmd_img = [
        sys.executable,
        os.path.join(ANALYSIS_SCRIPTS_DIR, "image_analysis.py"),
        "--image", img_path,
        "--output", img_out
        ]

        try:
            subprocess.run(cmd_img, check=True)

        # Fallback if script ignored --output (legacy behavior)
            legacy_candidate_img = os.path.join(
            person_dir, "analysis-p01", sample, "image_scores.json"
            )
            if not os.path.exists(img_out) and os.path.exists(legacy_candidate_img):
                os.makedirs(os.path.dirname(img_out), exist_ok=True)
                os.replace(legacy_candidate_img, img_out)

        except subprocess.CalledProcessError as e:
            flash(f"Image analysis failed: {e}", "error")

    results = {}
    if os.path.exists(text_out):
        with open(text_out) as f:
            results["text_scores"] = json.load(f)

    if os.path.exists(img_out):
        with open(img_out) as f:
            results["image_scores"] = json.load(f)

    if results:
        results_col.update_one(
            {"person_id": person, "sample_id": sample},
            {"$set": results},
            upsert=True
        )

    flash("Analysis complete.", "success")
    return redirect(url_for("index", person=person, sample=sample))



if __name__ == "__main__":
    app.run(debug=True)
