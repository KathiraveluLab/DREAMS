# DREAMS

**Digital Reckoning of Emotional and Affective Mood States**

DREAMS is an analytics platform that converts **image + caption data** into **time-aware emotional insights** for research and care.

---

## What It Does

From raw inputs, DREAMS produces:

* Emotion predictions (face + text)
* Scene/context classification
* Time-ordered emotion timelines
* Episode segmentation (meaningful phases)
* Context-aware proximity analysis

> In short: it turns experiences into structured emotional trajectories.

---

## Demo

**Input**

* Image: person sitting in a park
* Caption: "felt calm but exhausted"
* Timestamp: `2026-02-10 17:30`

**Output**

```json
{
  "emotion": "calm",
  "confidence": 0.81,
  "secondary_emotion": "fatigue",
  "scene": "park",
  "episode_id": 3,
  "timeline_position": "Day 12",
  "context_tag": "outdoor_recovery"
}
```

**Pipeline (high-level)**

1. Face emotion + caption sentiment
2. Scene classification (Places365)
3. Timeline building
4. Episode segmentation
5. Proximity analysis

---

## Architecture

```
Input (Image + Caption)
        ↓
ML Inference (Face, Text, Scene)
        ↓
Analytics (Timeline, Episodes, Proximity)
        ↓
Outputs (JSON / Visualization-ready)
```

---

## Repository

* `dreamsApp/` core analytics
* `ml/` inference utilities
* `tests/` unit + integration tests
* `data_integrity/` schema & time validation
* `dream-integration/` integration scripts
* `location_proximity/` spatial analysis
* `docs/` architecture & test plans

---

## Getting Started

```bash
git clone https://github.com/KathiraveluLab/DREAMS.git
cd DREAMS
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python analytics_server.py
```

**Expected**

* Server starts
* Models load
* Pipeline ready for requests

---

## Example Usage

```python
from dreamsApp.timeline_builder import build_timeline

data = [{
  "emotion": "happy",
  "timestamp": "2026-01-01 10:00",
  "scene": "home"
}]

print(build_timeline(data))
```

---

## Core Modules

* `emotion_timeline.py` → ordered trajectories from timestamps
* `episode_segmentation.py` → groups into phases (e.g., stress/recovery)
* `temporal_narrative_graph.py` → links events & transitions
* `emotion_proximity.py` → context closeness (time + place)
* `timeline_builder.py` → visualization-ready structures

---

## Workflow

1. Collect consented data
2. Validate (`data_integrity`)
3. Run inference (emotion + scene + sentiment)
4. Build timelines & episodes
5. Visualize and interpret

---

## Limitations

* Sensitive to image quality/lighting
* Emotion labels miss nuance
* Scene classification ambiguity
* Not a clinical tool

---

## Ethics

* Use only consented data
* Treat outputs as decision support
* Interpret with human context

---

## Contributing

**Beginner**

* Tests, docs, validation fixes

**Intermediate**

* Timeline optimization
* Episode logic improvements

**Advanced**

* Emotion fusion (face + text)
* Real-time pipeline
* Proximity algorithms

---

## Tests

```bash
pytest
pytest tests/test_timeline.py
```

---

## Docs

* `ARCHITECTURE.md`
* `SETUP.md`
* `docs/TEST_PLAN.md`

---

## Why It Matters

DREAMS focuses on **where, when, and how emotions change**, not just what they are—making it useful for real-world recovery analysis.

---

## Acknowledgment

An ongoing effort toward **context-aware, humane analytics**. Contributions welcome.
