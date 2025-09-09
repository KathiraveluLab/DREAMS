# Contributing to DREAMS Multimodal Emotion & Sentiment Analysis  

Thank you for your interest in contributing! 🙌  
This project is the groundwork for multimodal (audio, text, image) emotion and sentiment analysis with a structured pipeline and visualization.  

We welcome improvements, fixes, and feature enhancements. Please follow the setup and usage steps carefully before contributing.  

---

##  Setup Instructions  

1. **Fork and Clone**  
    - Fork this repository to your GitHub account.  
    - Clone your fork locally:  
     ```bash
     git clone https://github.com/<your-username>/dream-integration.git
     cd dream-integration
     ```

2. **Create Virtual Environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

3. **Install Requirements**
     ```bash
     pip install -r requirements.txt
     ```
4. **Run Local Setup**

    Navigate to the app directory and start Flask:

    ```bash
    cd app
    python app.py
     ```

## Usage Instructions

1. Place your data in the following structure:
    ```text
    data/
        person-01/
            sample-01/
                clip-01.mp3
                image-01.jpg
                description-01.txt
    ```
2. Run Modules Individually (to verify setup):
    - Audio → Text (Whisper):
    ```bash
    python analysis/transcribe_and_save.py data/person-01/sample-01/clip-01.mp3
     ```
     **Note: A transcript.txt file should appear inside the sample folder.**


     - Text Emotion Analysis (DistilBERT):
     ```bash
    python analysis/text_analysis.py --transcript data/person-01/sample-01/transcript.txt \
                                 --description data/person-01/sample-01/description-01.txt \
                                 --output data/person-01/analysis-p01/sample-01/text_scores.json
     ```
    **Note: text_scores.json should be created with proper emotion scores for the description and transcript both.**



     - Image Emotion Analysis (DeepFace):
     ```bash
    python analysis/image_analysis.py --image data/person-01/sample-01/image-01.jpg \
                                  --output data/person-01/analysis-p01/sample-01/image_scores.json
     ```
    **Note: image_scores.json should be created successfully.**


3. Once all modules generate their outputs, run the web app again:
     ```bash
    python app/app.py
     ```
    **Note: The setup is complete now, you should now see all results (transcript + scores + visualizations) in the browser.**


## Contributing Guidelines  

- Please read the Guidelines.md file carefully before contributing.  
- Keep commits meaningful and modular (e.g., fix: improve transcript path handling).  
- Always test your changes locally before submitting a Pull Request.  
- Ensure the existing folder structure is not broken.  
- Open an issue if you are proposing a change or a new feature.  



