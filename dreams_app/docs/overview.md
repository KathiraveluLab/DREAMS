# DREAMS Overview

## What is DREAMS?

**DREAMS** (Documenting Recovery Experiences through Emotion-Aware Multimedia Stories) is an open-source platform for analyzing how individuals document their recovery and emotional journeys through multimedia content—specifically, images and captions.

It is designed to support researchers, clinicians, and community organizations by providing tools for sentiment analysis, temporal tracking, and thematic clustering. DREAMS transforms narrative data into structured insights that can inform research, therapy, and outreach.

---

## Motivation

Recovery is a deeply personal and emotional process. Individuals often share images and stories that capture moments of transformation, struggle, and healing. However, most existing systems treat such data as static or qualitative-only, lacking tools for:

- Tracking emotional changes over time
- Understanding recurring themes or locations
- Quantifying narrative structure

DREAMS addresses this by providing a modular infrastructure to analyze and interpret these personal narratives at scale, while respecting context, ethics, and sensitivity.

---

## Intended Audience

DREAMS is designed for professionals and contributors working at the intersection of mental health, research, and technology, including:

- **Researchers** in behavioral health, digital storytelling, and affective computing
- **Clinicians and psychologists** seeking tools for patient reflection and progress tracking
- **Community organizations and social workers** supporting recovery documentation
- **Developers and data scientists** building AI for social good and narrative modeling

---

## Core Capabilities

- **Multimedia Entry Ingestion**: Users upload images and write captions to document key life moments.
- **Sentiment Analysis**: Captions are processed through transformer-based models to extract emotional tone.
- **Time-Aware Tracking**: Emotional states are indexed over time to reveal patterns and trajectories.
- **Keyword Clustering**: Recurring terms and concepts are grouped to surface thematic structures.
- **Thematic Analysis (LLM-based)**: Optional integration with large language models to summarize user journeys.

---

## Use Cases

- Tracking emotional recovery over time in patients undergoing therapy or addiction recovery
- Analyzing collective trends in emotional expression across groups or regions
- Identifying emotional triggers or supports based on places, people, or concepts
- Generating anonymized datasets for public health and social science research

---

## Design Principles

- **Ethical AI**: Models are fine-tuned with domain-relevant data and designed to minimize misclassification in sensitive contexts.
- **Privacy-Aware**: All data handling practices prioritize user control, anonymization, and transparency.
- **Extensibility**: Built with modular APIs and open standards to allow easy integration into larger platforms like Beehive or custom research workflows.
- **Open Collaboration**: Developed in the open with opportunities for research validation and community-driven extensions.

---

## Relation to Other Projects

DREAMS is part of the [Beehive](https://github.com/kathiraveluLab/beehive) ecosystem, a broader open-source initiative focused on participatory storytelling and data justice. It integrates with components such as:

- **Beehive Frontend/Backend**: For user story collection and visualization
- **DistilBERT**: For initial sentiment analysis
- **Gemini / LLM APIs**: For thematic interpretation

---

## Citation and Acknowledgment

This platform is under active development and is part of ongoing academic research.



