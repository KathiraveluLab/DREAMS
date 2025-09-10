# DREAM Integration: Multimodal Emotion & Sentiment Analysis

This project lays the complete groundwork — a strong and functional base variant — for multimodal sentiment and emotion analysis. It processes image, audio, and text data while treating time and ordering as key components. Time plays an important role in connecting and understanding the flow of these inputs. Rather than viewing each input in isolation, the system captures the sequence of events, helping us understand how earlier moments shape emotional responses in later ones. Data mining is used to detect patterns and trends by organizing multimodal emotion data in a timeline per person, allowing us to trace how feelings evolve over time. This approach builds a richer narrative from lived experiences, aligning with Beehive’s goal of creating time-aware, emotion-driven digital memory systems.

## Current Status

- Project structure, sample data folders, virtual environment, and requirements file created.  
- Audio transcript generation using [OpenAI Whisper](https://github.com/openai/whisper).  
- Text emotion scoring using DistilBERT; results stored in structured format (`analysis-*/text_scores.json`).  
- Image sentiment analysis integrated using DeepFace.  
- Unified **analysis pipeline** in Flask:  
  - Transcription → Text analysis → Image analysis.  
  - Handles missing transcripts, avoids overwriting, ensures consistent output storage.  
- **Radar chart visualization** added for comparing text vs image emotions.  
- Progress bars and collapsible raw JSON views included in UI.  

## Tools Used

- Whisper (for audio transcription)  
- HuggingFace Transformers + DistilBERT (for text emotion scoring)  
- Torch (PyTorch backend)  
- DeepFace (for facial emotion/sentiment analysis)  
- Flask + Chart.js (for web UI & visualization)  
- Python 3.x, Virtualenv  

## Project Folder Highlights

- `dream-integration/analysis/` → Modular scripts for analyzing text, audio, and image inputs.  
- `dream-integration/data/person-01/analysis-*/` → Per-user, per-sample emotion scores stored in JSON format.  

## Future Scope

- Extending the current end-to-end tested implementation (single-person multi-sample) to handle **multi-person multi-sample** analysis.  
- Visualization can be further enriched into **timeline-based analysis**, to track emotional progression over time. This will be more meaningful when a larger dataset becomes available in the future.  

## Architecture

The workflow is designed around three modalities: **Audio, Text, and Image**.
All input data is scanned from the data/ directory and processed step by step through specialized models. Results are aggregated and displayed via a simple web UI with Chart.js visualizations.

### High-Level Architecture Diagram

```mermaid
flowchart LR
    subgraph User
        U1[Clicks Start Analysis]
    end

    subgraph Frontend[Frontend UI with Chart.js]
        F1[Trigger Analysis Request]
        F2[Show Raw JSON]
        F3[Show Charts - Radar & Bar]
    end

    subgraph Backend[Backend / API]
        B1[Scan data folder for files]
        B2[Aggregate Results]
    end

    subgraph Models[ML / DL Models]
        M1[Whisper - Audio → Transcript]
        M2[DistilBERT - Analyze Texts]
        M3[DeepFace - Analyze Image]
    end

    U1 --> F1 --> B1
    B1 -->|Audio| M1 --> M2
    B1 -->|Description| M2
    B1 -->|Image| M3

    M2 -->|Text Scores JSON| B2
    M3 -->|Image Scores JSON| B2
    B2 --> F2
    B2 --> F3
```
### Workflow (Step-by-Step)

Below is a step-by-step pipeline of how the system works when the user starts analysis:
```mermaid
flowchart TD
    A[User clicks Start Analysis] --> B[Backend scans data folder]
    B --> C[Audio File]
    B --> D[Description File]
    B --> E[Image File]

    C --> F[Whisper: Transcribe Audio]
    F --> G[DistilBERT: Analyze Transcript]
    D --> H[DistilBERT: Analyze Description]
    G --> I[Text Score JSON]
    H --> I

    E --> J[DeepFace: Analyze Image]
    J --> K[Image Score JSON]

    I --> L[Aggregator merges JSON]
    K --> L

    L --> M[Frontend UI]
    M --> N[Show Raw JSON]
    M --> O[Show Charts - Radar and Bar]
```

## Setup & Contributing  
For setup instructions and contributing guidelines, please go through the dedicated file.  
(Currently working on it, will be added soon.)

## License  
This project is licensed under the MIT License.  See the [LICENSE](/LICENSE) file for more details.