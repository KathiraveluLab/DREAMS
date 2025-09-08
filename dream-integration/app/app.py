import os
import json
import glob
import subprocess
import sys
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "dev"

# ---- PATHS ----
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  #dream-integration
DATA_DIR = os.path.join(BASE_DIR, "data")
ANALYSIS_SCRIPTS_DIR = os.path.join(BASE_DIR, "analysis")

ALLOWED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

def list_persons():
    """Return sorted list of person folder names under data/."""
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

def list_samples(person):
    pdir = os.path.join(DATA_DIR, person)
    if not os.path.isdir(pdir):
        return []
    return sorted([d for d in os.listdir(pdir) if os.path.isdir(os.path.join(pdir, d)) and d.startswith("sample")])

def find_image(sample_dir):
    for f in os.listdir(sample_dir):
        _, ext = os.path.splitext(f)
        if ext.lower() in ALLOWED_IMG_EXTS:
            return f
    return None

def find_transcript(sample_dir):
    # For both clip-*.txt and transcript-*.txt
    p = os.path.join(sample_dir, "transcript.txt")
    if os.path.isfile(p):
        return os.path.basename(p)

    cand = sorted(glob.glob(os.path.join(sample_dir, "transcript-*.txt")))
    if cand:
        return os.path.basename(cand[0])

    cand2 = sorted(glob.glob(os.path.join(sample_dir, "clip-*.txt")))
    if cand2:
        return os.path.basename(cand2[0])

    return None

def find_audio(sample_dir):
    """Find any .mp3 or .wav file in sample_dir. Returns basename or None."""
    cand = sorted(glob.glob(os.path.join(sample_dir, "*.mp3")) +
                  glob.glob(os.path.join(sample_dir, "*.wav")))
    return os.path.basename(cand[0]) if cand else None

def find_description(sample_dir):
    cand = sorted(glob.glob(os.path.join(sample_dir, "description*.txt")))
    return os.path.basename(cand[0]) if cand else None

def get_analysis_dir(person, sample):
    """
    Preferred: data/<person>/analysis-<person>/<sample>/
    Backward-compat: if data/<person>/analysis-p01 exists, use that instead.
    """
    person_dir = os.path.join(DATA_DIR, person)
    legacy = os.path.join(person_dir, "analysis-p01", sample)
    if os.path.isdir(os.path.join(person_dir, "analysis-p01")):
        os.makedirs(legacy, exist_ok=True)
        return legacy

    modern = os.path.join(person_dir, f"analysis-{person}", sample)
    os.makedirs(modern, exist_ok=True)
    return modern

def read_text(path):
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def read_json(path):
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def media_url(rel_path):
    # serve any file from data/ through /media/<path>
    return url_for("serve_media", path=rel_path)

@app.route("/media/<path:path>")
def serve_media(path):
    """Serve files from data/ for previews (images, etc.)."""
    safe_root = DATA_DIR
    directory = os.path.dirname(path)
    filename = os.path.basename(path)
    return send_from_directory(os.path.join(safe_root, directory), filename)

@app.route("/", methods=["GET", "POST"])
def index():
    persons = list_persons()
    if not persons:
        return "No persons found under ./data/. Create data/person-01/sample-01 and add files.", 200

    # selection (defaults to first)
    person = request.values.get("person", persons[0])
    samples = list_samples(person)
    if not samples:
        return f"No samples found for {person}. Create data/{person}/sample-01.", 200
    sample = request.values.get("sample", samples[0])

    # build paths
    sample_dir = os.path.join(DATA_DIR, person, sample)
    img_name = find_image(sample_dir)
    transcript_name = find_transcript(sample_dir)
    description_name = find_description(sample_dir)

    # relative paths (for /media)
    img_rel = f"{person}/{sample}/{img_name}" if img_name else None
    transcript_rel = f"{person}/{sample}/{transcript_name}" if transcript_name else None
    description_rel = f"{person}/{sample}/{description_name}" if description_name else None

    # analysis output dir + expected jsons
    out_dir = get_analysis_dir(person, sample)
    text_json = os.path.join(out_dir, "text_scores.json")
    image_json = os.path.join(out_dir, "image_scores.json")

    context = {
        "persons": persons,
        "samples": samples,
        "selected_person": person,
        "selected_sample": sample,
        "img_url": media_url(img_rel) if img_rel else None,
        "transcript_text": read_text(os.path.join(DATA_DIR, transcript_rel)) if transcript_rel else "",
        "description_text": read_text(os.path.join(DATA_DIR, description_rel)) if description_rel else "",
        "text_scores": read_json(text_json),
        "image_scores": read_json(image_json),
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
    """
    person = request.form["person"]
    sample = request.form["sample"]

    sample_dir = os.path.join(DATA_DIR, person, sample)
    person_dir = os.path.join(DATA_DIR, person)
    out_dir = get_analysis_dir(person, sample)
    os.makedirs(out_dir, exist_ok=True)

    # Find files
    img = find_image(sample_dir)
    transcript = find_transcript(sample_dir)
    description = find_description(sample_dir)

    # Transcribe audio if transcript missing
    if not transcript:
        audio_basename = find_audio(sample_dir)
        if audio_basename:
            audio_path = os.path.join(sample_dir, audio_basename)
            before_txt = set(glob.glob(os.path.join(sample_dir, "*.txt")))

            try:
                cmd_audio = [
                    sys.executable,
                    os.path.join(ANALYSIS_SCRIPTS_DIR, "transcribe_and_save.py"),
                    audio_path
                ]
                subprocess.run(cmd_audio, check=True)

                after_txt = set(glob.glob(os.path.join(sample_dir, "*.txt")))
                new_txt = sorted(list(after_txt - before_txt))

                # ignore description files
                new_txt_filtered = [p for p in new_txt if not os.path.basename(p).lower().startswith("description")]

                chosen = None
                if new_txt_filtered:
                    chosen = new_txt_filtered[0]
                else:
                    fallback = find_transcript(sample_dir)
                    if fallback:
                        chosen = os.path.join(sample_dir, fallback)

                if chosen:
                    target = os.path.join(sample_dir, "transcript.txt")
                    if os.path.abspath(chosen) != os.path.abspath(target):
                        if os.path.exists(target):
                            os.remove(target)
                        os.replace(chosen, target)
                    transcript = "transcript.txt"
                else:
                    flash("Transcription finished but no transcript file detected.", "error")

            except subprocess.CalledProcessError as e:
                flash(f"Audio transcription failed: {e}", "error")

    # Paths
    img_path = os.path.join(sample_dir, img) if img else None
    transcript_path = os.path.join(sample_dir, transcript) if transcript else None
    description_path = os.path.join(sample_dir, description) if description else None

    # Text analysis - NEW: Check if output already exists
    text_out = os.path.join(out_dir, "text_scores.json")
    if not os.path.exists(text_out) and (transcript_path or description_path):
        cmd_text = [
            sys.executable,
            os.path.join(ANALYSIS_SCRIPTS_DIR, "text_analysis.py"),
            "--output", text_out
        ]
        if transcript_path:
            cmd_text += ["--transcript", transcript_path]
        if description_path:
            cmd_text += ["--description", description_path]

        try:
            subprocess.run(cmd_text, check=True)
            # Fallback if script ignored --output (check inside sample folder too)
            legacy_candidate = os.path.join(person_dir, "analysis-p01", sample, "text_scores.json")
            if not os.path.exists(text_out) and os.path.exists(legacy_candidate):
                os.makedirs(os.path.dirname(text_out), exist_ok=True)
                os.replace(legacy_candidate, text_out)
        except subprocess.CalledProcessError as e:
            flash(f"Text analysis failed: {e}", "error")

    # Image analysis - ensure output lands in correct folder
    img_out = os.path.join(out_dir, "image_scores.json")
    if img_path:
        # Only run if file doesn't already exist
        if not os.path.exists(img_out):
            cmd_img = [
                sys.executable,
                os.path.join(ANALYSIS_SCRIPTS_DIR, "image_analysis.py"),
                "--image", img_path,
                "--output", img_out
            ]
            try:
                subprocess.run(cmd_img, check=True)
                # Fallback: if script ignored --output
                legacy_candidate_img = os.path.join(person_dir, "analysis-p01", sample, "image_scores.json")
                if not os.path.exists(img_out) and os.path.exists(legacy_candidate_img):
                    os.makedirs(os.path.dirname(img_out), exist_ok=True)
                    os.replace(legacy_candidate_img, img_out)
            except subprocess.CalledProcessError as e:
                flash(f"Image analysis failed: {e}", "error")

    flash("Analysis complete.", "success")
    return redirect(url_for("index", person=person, sample=sample))

if __name__ == "__main__":
    # run:  python app/app.py
    app.run(debug=True)