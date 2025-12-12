# DREAMS

Digitization for Recovery: Exploring Arts with Mining for Societal well-being.

DREAMS is an extension of the Beehive project, focused on exploring time and ordering across photo memories to better understand personal recovery journeys. The goal is to build tools that help track and analyze visual narratives over time using data mining and intelligent processing.

## Current Progress

- Set up core infrastructure using Flask and Hugging Face models.
- Implemented a basic **Caption Sentiment Analysis API** to classify emotional tone in user-submitted captions.
- Integrating this API into Beehive to capture sentiment when users upload photos.
- Exploring time-based data structuring and narrative analysis features.

### [View the API Module](./dreamsApp/README.md)

## Repositories

- Beehive: [github.com/KathiraveluLab/beehive](https://github.com/KathiraveluLab/Beehive)
- DREAMS: [github.com/KathiraveluLab/DREAMS](https://github.com/KathiraveluLab/DREAMS)


## Repository Structure

    DREAMS/
    ├── dreamsApp/
    │   ├── __init__.py                #        App factory
    │   ├── captionSentiments.py       # API logic and model loading
    |   ├── README.md
    ├──tests/
    |  ├──test_sentiment.py
    ├──pytest.ini
    ├──README.md
    ├──requirements.txt
 
## Installation and Setup

```bash
# 1. Clone the repository
git clone https://github.com/KathiraveluLab/DREAMS.git
cd DREAMS

# 2. (Optional but recommended) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install the required dependencies
pip install -r requirements.txt

# 4. Run tests to verify everything is working
pytest

# 5. Start the Flask server in debug mode
flask --app dreamsApp run --debug
```

More coming soon!
