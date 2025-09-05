# Sentiment Analysis

We use the Twitter-RoBERTa model, which was fine-tuned on Twitter data collected during the COVID-19 period. This model was chosen because:

- It performs well on short, informal, and social media-style text, similar to the captions received from Beehive.

- We currently lack sufficient domain-specific data to fine-tune a custom sentiment model.

## API: **/api/upload**

This endpoint accepts a POST request from Beehive, processes the data (caption + image), performs sentiment analysis, generates an image caption, and stores the result in the database.

### Example Request
```json
{
  "user_id": "101",
  "caption": "Seventy-five days in. I never thought I’d make it this far. Super proud.",
  "timestamp": "2025-08-06T18:30:00",
  "image": "family.jpeg"
}
```
### Example Response:
```json
{
    "caption": "Seventy-five days in. I never thought I’d make it this far. Super proud.",
    "generated_caption": "arafed family with two children posing for a picture",
    "image_path": "/home/DREAMS/DREAMS/dreamsApp/images/family.jpeg",
    "message": "Post created successfully",
    "post_id": "68ac0594891111fd92300f17",
    "sentiment": {
        "label": "positive",
        "score": 0.9822999835014343
    },
    "timestamp": "Wed, 06 Aug 2025 18:30:00 GMT",
    "user_id": "101"
}
```
## How It works
- The API calls get_image_caption_and_sentiment() (in utils/sentiment.py).

- This function:

    Runs sentiment analysis on the provided text caption using Twitter-RoBERTa.

    Generates an image caption using Salesforce BLIP.

    Combines both results into a structured JSON object.

- The processed data is stored in MongoDB for further use and analysis.

