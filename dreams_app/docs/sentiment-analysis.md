# Sentiment Analysis

We use the Twitter-RoBERTa model, which was fine-tuned on Twitter data collected during the COVID-19 period. This model was chosen because:

- It performs well on short, informal, and social media-style text, similar to the captions received from Beehive.

- We currently lack sufficient domain-specific data to fine-tune a custom sentiment model.

## API: **/api/upload**

This endpoint accepts a `POST` request from Beehive as `multipart/form-data` (not JSON) and requires a logged-in session. It validates the image, processes the data (caption + image), performs sentiment analysis, generates an image caption, and stores the result in the database.

Form fields: `user_id`, `caption` and `image` (a file) are required. `timestamp` is optional, in ISO 8601 format; when omitted, the current UTC time is used. Missing fields, an invalid timestamp, an unsupported file extension or a file that is not a valid image return `400` with an `error` message.

### Example Request
```bash
curl -X POST http://127.0.0.1:5001/api/upload \
  -F "user_id=101" \
  -F "caption=Seventy-five days in. I never thought I’d make it this far. Super proud." \
  -F "timestamp=2025-08-06T18:30:00" \
  -F "image=@family.jpeg"
```
### Example Response (`201 Created`):
```json
{
    "caption": "Seventy-five days in. I never thought I’d make it this far. Super proud.",
    "generated_caption": "arafed family with two children posing for a picture",
    "image_path": "/home/DREAMS/DREAMS/dreams_app/images/3f2a9c0e5b7d4e1f8a6b2c4d9e0f1a2b_family.jpeg",
    "message": "Post created successfully",
    "post_id": "68ac0594891111fd92300f17",
    "sentiment": {
        "label": "positive",
        "score": 0.9822999835014343
    },
    "timestamp": "2025-08-06T18:30:00+00:00",
    "user_id": "101"
}
```
## How It works
- The core sequence calls `get_image_caption_and_sentiment()` (in `core/sentiment.py`).

- This function:

    Runs sentiment analysis on the provided text caption using Twitter-RoBERTa.

    Generates an image caption using Salesforce BLIP.

    Combines both results into a structured JSON object.

- The processed data is stored in MongoDB for further use and analysis.

