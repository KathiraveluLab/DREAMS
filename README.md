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
├── dreamsApp/                  # Main application package
│   ├── app/                    # Flask app package (app factory + blueprints)
│   │   ├── __init__.py         # create_app() factory
│   │   ├── config.py           # App configuration
│   │   ├── models.py           # Database models
│   │   ├── auth.py             # Authentication routes
│   │   │
│   │   ├── ingestion/          # Image ingestion & processing
│   │   │   ├── __init__.py
│   │   │   └── routes.py
│   │   │
│   │   ├── dashboard/          # Dashboard & analytics views
│   │   │   ├── __init__.py
│   │   │   └── main.py
│   │   │
│   │   └── utils/              # Core ML / NLP utilities
│   │       ├── sentiment.py    # Caption sentiment analysis
│   │       ├── keywords.py     # Keyword extraction
│   │       ├── clustering.py   # Keyword clustering (HDBSCAN)
│   │       └── llms.py         # LLM (Gemini) integration
│   │
│   └── docs/                   # Project documentation
│
├── data_integrity/             # Data validation utilities
├── location_proximity/         # Location-based analysis (future)
├── dream-integration/          # Integration & experimental code
├── tests/                      # Unit and integration tests
│
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Pytest configuration
└── README.md                   # Project documentation

 
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
flask --app "dreamsApp.app:create_app()" run --debug
```

More coming soon!
