"""
Text-based Sentiment Analysis for DREAMS

Analyzes text descriptions (photo captions) to estimate positive/negative/neutral
sentiment percentages. Uses the transformers pipeline with
j-hartmann/emotion-english-distilroberta-base model, or falls back to keyword-based
analysis if the model is unavailable.

Emotion mapping (7-class → 3-class):
    joy, surprise      → positive
    neutral            → neutral
    anger, disgust,
    fear, sadness      → negative
"""

from typing import Dict
import re

# Lazy-load transformers pipeline
_sentiment_pipeline = None
_pipeline_loaded = False
_pipeline_error = None


def _clean_text(text: str) -> str:
    """Strip emojis, hashtags, and excess whitespace for cleaner model input."""
    # Remove emoji characters (Unicode emoji ranges)
    text = re.sub(
        r'[\U0001F600-\U0001F64F'   # emoticons
        r'\U0001F300-\U0001F5FF'     # symbols & pictographs
        r'\U0001F680-\U0001F6FF'     # transport & map
        r'\U0001F1E0-\U0001F1FF'     # flags
        r'\U00002702-\U000027B0'
        r'\U000024C2-\U0001F251'
        r'\U0000200d'                # zero-width joiner
        r'\U0000fe0f]+', ' ', text)
    # Remove hashtags but keep the word
    text = re.sub(r'#(\w+)', r'\1', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

EMOTION_TO_CATEGORY = {
    'anger': 'negative',
    'disgust': 'negative',
    'fear': 'negative',
    'joy': 'positive',
    'sadness': 'negative',
    'surprise': 'positive',   # Align with perceptual model's mapping for consistency
    'neutral': 'neutral',
}

# Keyword sets for fallback analysis
POSITIVE_WORDS = {
    'happy', 'joy', 'amazing', 'good', 'great', 'love', 'lighter', 'grateful',
    'laughs', 'better', 'wonderful', 'beautiful', 'fantastic', 'awesome',
    'goodvibes', 'coffeetime', 'smile', 'excited', 'delighted', 'bright',
    'cheerful', 'warm', 'fun', 'enjoy', 'glad', 'pleased', 'incredible',
}
NEGATIVE_WORDS = {
    'sad', 'heavy', 'overwhelmed', 'bad', 'terrible', 'awful', 'depressed',
    'anxious', 'worry', 'hurt', 'pain', 'cry', 'mentalhealth', 'struggle',
    'fear', 'angry', 'disgust', 'grief', 'lonely', 'stressed', 'exhausted',
}
NEUTRAL_WORDS = {
    'daily', 'routine', 'normal', 'regular', 'usual', 'commute', 'grind',
    'moving', 'forward', 'keepgoing', 'nothing', 'special', 'walk', 'working',
    'day', 'getting', 'putting', 'foot', 'front',
}


def _load_pipeline():
    """Lazy-load the transformers sentiment pipeline."""
    global _sentiment_pipeline, _pipeline_loaded, _pipeline_error
    if _pipeline_loaded:
        return _sentiment_pipeline
    _pipeline_loaded = True
    try:
        from transformers import pipeline
        _sentiment_pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
        )
        print("✅ Text sentiment pipeline loaded (j-hartmann/emotion-english-distilroberta-base)")
        return _sentiment_pipeline
    except Exception as e:
        _pipeline_error = str(e)
        print(f"⚠️  Text sentiment pipeline not available: {e}. Using keyword fallback.")
        return None


def _keyword_fallback(text: str) -> Dict:
    """Simple keyword-based sentiment analysis fallback."""
    words = set(
        re.sub(r"[#.,!?'\"]", " ", text.lower()).split()
    )

    pos_count = len(words & POSITIVE_WORDS)
    neg_count = len(words & NEGATIVE_WORDS)
    neu_count = len(words & NEUTRAL_WORDS)
    total = pos_count + neg_count + neu_count

    if total == 0:
        return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}

    positive = pos_count / total
    negative = neg_count / total
    neutral = neu_count / total
    return {
        'positive': round(positive, 4),
        'negative': round(negative, 4),
        'neutral': round(neutral, 4),
    }


def analyze_text_sentiment(text: str) -> Dict:
    """
    Analyze text for positive/negative/neutral sentiment.

    Pipeline:
        text description → ML model → positive % | negative % | neutral %

    Returns:
        Dict with 'positive', 'negative', 'neutral' (0-1 floats),
        plus metadata: uncertainty_margin, dominant_emotion, notes, etc.
    """
    pipe = _load_pipeline()

    # Clean text for better model accuracy
    cleaned = _clean_text(text)

    if pipe is not None:
        try:
            results = pipe(cleaned[:512])  # Truncate to model max
            emotions = {r['label']: r['score'] for r in results[0]}

            positive = 0.0
            negative = 0.0
            neutral = 0.0

            for emotion, score in emotions.items():
                category = EMOTION_TO_CATEGORY.get(emotion, 'neutral')
                if category == 'positive':
                    positive += score
                elif category == 'negative':
                    negative += score
                else:
                    neutral += score

            dominant = max(emotions, key=emotions.get)

            return {
                'positive': round(positive, 4),
                'negative': round(negative, 4),
                'neutral': round(neutral, 4),
                'uncertainty_margin': round(
                    min(0.20, 0.05 + 0.15 * (1.0 - max(positive, negative, neutral))), 4
                ),
                'detailed_emotions': {k: round(v, 4) for k, v in emotions.items()},
                'dominant_emotion': dominant,
                'dominant_confidence': round(emotions[dominant], 4),
                'model': 'j-hartmann/emotion-english-distilroberta-base',
                'source': 'text',
                'notes': f'Text analysis: {dominant} detected ({emotions[dominant]:.1%} confidence).',
                'disclaimer': 'Sentiment estimated from text description. Results are probabilistic.',
            }
        except Exception as e:
            print(f"Pipeline inference error, using fallback: {e}")

    # Keyword-based fallback
    result = _keyword_fallback(cleaned)
    dominant = max(['positive', 'negative', 'neutral'], key=lambda k: result[k])
    result.update({
        'uncertainty_margin': 0.15,
        'detailed_emotions': {},
        'dominant_emotion': dominant,
        'dominant_confidence': round(result[dominant], 4),
        'model': 'keyword-fallback',
        'source': 'text',
        'notes': f'Keyword-based analysis: {dominant} ({result[dominant]:.1%}).',
        'disclaimer': 'Sentiment estimated via keyword matching. Transformer model unavailable.',
    })
    return result
