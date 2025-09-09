# Contribution Guidelines

Before you begin contributing, kindly take a moment to go through the [Contributing.md](Contributing.md) file.  
It will help you understand the setup, usage, and contribution process more smoothly.

---

## Project Directory Structure

This project is **path-sensitive**. The analysis pipeline, data storage, and Flask UI depend on the folder organization.  
Please do not move or rename critical directories without a solid backing.

```mermaid
graph TD;
    A[dream-integration/]

    %% Analysis
    A --> B[analysis/]
    B --> B1[text_analysis.py]
    B --> B2[image_analysis.py]
    B --> B3[transcribe_and_save.py]

    %% App
    A --> C[app/]
    C --> C1[app.py]
    C --> C2[templates/ → index.html]
    C --> C3[static/css/ → styles.css]

    %% Data
    A --> D[data/]
    D --> D1[person-01/]
    D1 --> D11[sample-01/]
    D11 --> D111[clip-01.mp3]
    D11 --> D112[transcript.txt]
    D11 --> D113[description-01.txt]
    D11 --> D114[image-01.png]

    D1 --> D12[analysis-p01/]
    D12 --> D121[text_scores.json]
    D12 --> D122[image_scores.json]

    %% Root files
    A --> E[README.md]
    A --> F[Contributing.md]
    A --> G[Guidelines.md]
    A --> H[requirements.txt]
```

---

## A Few Helpful Tips

- **Analysis-related changes** → keep inside `analysis/`
  - (e.g., add a new model wrapper here)
- **UI changes** → only inside `app/` (Flask, templates, CSS)
- **Data** → must follow the `data/<person>/<sample>/` format  
  - Transcripts must be named `transcript.txt`  
  - Outputs will be named `text_scores.json` and `image_scores.json`
- **Do not hardcode paths** — always keep them relative
- **Do not rename folders** — the pipeline depends on consistent names


