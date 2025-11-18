🚀 **DREAMS: Digitization for Recovery

Exploring Arts with Mining for Societal Well-Being**

DREAMS is an extension of the Beehive Project aimed at understanding personal recovery journeys through visual narratives. By combining art, memory, and machine intelligence, DREAMS explores how patterns across photo memories can reveal emotional trends, behavioral transitions, and subtle indicators of well-being.

This project experiments with advanced image–text analysis, temporal ordering, and narrative modeling to support mental-health-focused research with interpretable AI.

🌟 Project Vision

Modern digital archives (photos, captions, timelines) capture emotional stories that are rarely analyzed systematically. DREAMS aims to build:

Tools that automatically process visual diaries

Intelligent systems that detect emotional tone across time

Models that help researchers understand recovery trajectories

Art-driven insights for societal well-being

DREAMS supports the long-term mission of building responsible, research-grade AI for mental health contexts.

🔨 Current Progress

The current prototype is experimental and evolving rapidly.

✔️ Implemented

Flask-based core backend

Hugging Face caption sentiment classifier

Basic API for caption sentiment prediction

Tests validating the model behavior

🔄 In Progress

Integration with Beehive’s main platform

Time-aware narrative analysis

Improved temporal ordering of image sets

🧪 Experimental Directions

Multi-modal fusion (image + text)

Behavioral pattern mining

Emotional trajectory visualization

🚀 Getting Started

Set up DREAMS locally with the steps below.

1. Clone the repository
git clone https://github.com/KathiraveluLab/DREAMS.git
cd DREAMS

2. (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# OR
.\.venv\Scripts\activate         # Windows

3. Install dependencies
pip install -r requirements.txt


If the file is missing or empty (experimental phase), install core dependencies manually:

pip install flask transformers torch numpy

4. Run tests
pytest

5. Start the development server
flask --app dreamsApp run --debug

📁 Repository Structure
DREAMS/
├── dreamsApp/
│   ├── __init__.py                # Flask app factory
│   ├── captionSentiments.py       # HF model + sentiment API
│   ├── README.md
├── tests/
│   ├── test_sentiment.py
├── pytest.ini
├── requirements.txt
└── README.md

🧭 Roadmap (2025 → GSoC 2026)

This roadmap outlines potential growth directions for DREAMS, particularly for contributors and upcoming GSoC proposals.

Phase 1 — Infrastructure & Stability (Late 2025)

Improve folder structure & modularity

Add CI (GitHub Actions) for tests

Expand API responses (confidence, multi-label)

Add sample datasets for development

Improve documentation

Phase 2 — ML Architecture Upgrade (Early 2026)

Upgrade caption sentiment model to RoBERTa / DistilBERT

Add image emotion recognition (FER+, ViT, or EfficientNet)

Build a fusion model that considers both image + text

Add explainability (Grad-CAM, attention maps)

Improve inference pipeline for researcher use

Phase 3 — Temporal & Narrative Modeling (Pre-GSoC Submission)

Model emotional sequences over time

Cluster photos based on visual similarity + mood

Extract temporal story arcs

Build recovery-trajectory features

Add metadata analysis (time-of-day, location, etc.)

Phase 4 — GSoC 2026 Project Directions

Potential high-impact proposals include:

1️⃣ Multimodal Emotion Understanding

Image + caption + context → unified emotional prediction.

2️⃣ Timeline-Based Visual Narrative Engine

Automatically organizes and analyzes recovery journeys.

3️⃣ Interpretable AI for Mental Health

Explainable emotion & behavior analysis tools.

4️⃣ Integration with Beehive

Complete integration pipeline for real researcher usage.

5️⃣ Visual Recovery Dashboard

Charts, timelines, emotional arcs, and behavioral shifts.

🤝 Contributing

Contributions are welcome!
To participate:

Fork the repository

Create a feature branch

Commit your improvements

Open a Pull Request and describe the change

A full CONTRIBUTING.md guide will be added soon.

For major feature proposals, open an Issue first so we can discuss design direction.

📚 Related Repositories

Beehive Platform
https://github.com/KathiraveluLab/Beehive

DREAMS Project
https://github.com/KathiraveluLab/DREAMS

🎨 Acknowledgements

This project builds upon the efforts of GSoC 2025 contributors and the vision of the Beehive research initiative.
DREAMS continues to evolve as a collaborative space for meaningful, human-centered AI research.
