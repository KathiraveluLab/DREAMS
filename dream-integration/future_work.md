# Future Work: DREAM Integration

This document outlines the planned extensions for the DREAM Integration project.  
The current system is end-to-end tested for **local directory-based analysis**.  
Future work focuses on **database integration** and **Beehive interoperability**.

---

## 1. MongoDB Integration

### Current
- All inputs (audio, image, text) and outputs (JSON scores) are stored under the `data/` directory.
- Flask (`app.py`) scans the filesystem to list persons, samples, and their files.

### Planned
- Replace filesystem scanning with MongoDB queries:
  - `users` collection → list of persons.
  - `samples` collection → metadata + binary fields (GridFS for audio/image, plain text for transcripts/descriptions).
  - `results` collection → store analysis outputs (instead of `.json` files).

### Minimal Changes Needed
- Update helper functions (`list_persons`, `list_samples`, `find_image`, `find_transcript`) to query MongoDB instead of reading directories.
- Adapt `read_text` and `read_json` to fetch from MongoDB collections.
- Keep the Flask UI unchanged — it will continue to receive the same data from backend functions.

---

## 2. Beehive Integration

### Current
- Samples are organized per-person and per-sample (`sample-01`, `sample-02`, …).
- Analysis results (`text_scores.json`, `image_scores.json`) are already aligned with timeline-aware analysis.

### Planned
- Replace `data/` root with Beehive’s datastore (can be MongoDB-backed).
- Map Beehive metadata (timestamps, EXIF, tags, spatial data) into the DREAM pipeline.
- Extend visualization to show **timeline-based progression of emotions** across samples.

### Minimal Changes Needed
- Add a connector in `app.py` to fetch person/sample metadata from Beehive APIs.
- Ensure analysis output schema matches Beehive requirements.
- Enhance charts in `index.html` to support time-based plots (line charts or timeline visualizations).

---

## Why This is Feasible

The current system was deliberately structured around **clear person/sample separation and modular analysis scripts**.  
This makes switching from local files → MongoDB → Beehive a matter of updating the data source, while keeping the core pipeline and UI intact.

