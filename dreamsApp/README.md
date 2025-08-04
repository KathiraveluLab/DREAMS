# DREAMS

DREAMS is a Flask web application for uploading images and captions, performing sentiment analysis, extracting keywords, clustering, and visualizing user emotional journeys over time.

## Features

- **User Authentication** (Flask-Login)
- **Image & Caption Upload**
- **Sentiment Analysis** on captions
- **Keyword Extraction** and clustering
- **User Dashboard** with emotional journey plots and word clouds
- **MongoDB** backend for data storage

## Requirements

- Python 3.8+
- MongoDB
- [See `requirements.txt` for Python dependencies]

## Setup

1. **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/dreams.git
    cd dreams
    ```

2. **Create a virtual environment and activate it**
    ```bash
    python3 -m venv myenv
    source myenv/bin/activate
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Configure environment variables**

    Create a `.env` file or edit `dreamsApp/app/config.py` with your MongoDB URI and other settings:
    ```
    MONGO_URI=mongodb://localhost:27017/
    MONGO_DB_NAME=dreams_db
    SECRET_KEY=your_secret_key
    UPLOAD_FOLDER=uploads
    ```

5. **Run MongoDB**

    Make sure MongoDB is running locally or update the URI for your setup.

6. **Run the app**
    ```bash
    export FLASK_APP=dreamsApp
    export FLASK_ENV=development
    flask run
    ```

## Usage

- Register and log in.
- Upload images and captions.
- View your dashboard for emotional journey and keyword analysis.
- Admins can run clustering via `/run_clustering`.

## Project Structure

```
dreamsApp/
  app/
    __init__.py
    models.py
    auth/
    dashboard/
    ingestion/
    utils/
  requirements.txt
  README.md
```

## API Endpoints

- `POST /ingestion/upload` — Upload a post (image + caption)
- `GET /dashboard/` — Main dashboard
- `GET /dashboard/user/<user_id>` — User profile with analytics
- `GET /ingestion/run_clustering` — Run keyword clustering

## License

MIT License

---

*For questions or contributions, please open an issue or pull request.*