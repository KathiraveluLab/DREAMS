"""
Generate synthetic_dataset5.jsonl that exactly matches the real pipeline output schema.

Target schema based on real memory output:
{
  "caption": str | null,
  "caption_source": "user" | "generated" | null,
  "captured_at": ISO8601 with timezone,
  "category": str | null,
  "embeddings": {
    "caption": {"collection": "caption_embeddings", "dimensions": 384},
    "image":   {"collection": "image_embeddings",   "dimensions": 512}
  },
  "emotions": {
    "arousal": float | null,
    "chime": {"category": str, "confidence": float},
    "discrete": {anger, disgust, fear, joy, neutral, sadness, surprise},
    "dominant_emotion": str,
    "sentiment": {"label": str, "negative": float, "neutral": float, "positive": float},
    "valence": float | null
  },
  "generated_caption": str,
  "image_path": str,
  "is_duplicate": bool,
  "location": {
    "address": {"city": str, "country": str, "road": str, "state": str},
    "display_name": str,
    "latitude": float,
    "longitude": float,
    "place_type": str
  },
  "memory_id": 16-char hex string,
  "processing_status": "complete",
  "temporal": {
    "cyclical": {cos_dow, cos_hour, cos_month, sin_dow, sin_hour, sin_month},
    "day_of_week": int,
    "hour": int,
    "month": int,
    "recovery_day": float,
    "season": str,
    "time_of_day": str,
    "year": int
  },
  "user_id": str
}
"""

import json
import math
import random
import secrets
from datetime import datetime, timezone, timedelta

random.seed(42)

# ── helpers ──────────────────────────────────────────────────────────────────

def hex_id(n=16):
    return secrets.token_hex(n // 2)

def softmax(vals):
    e = [math.exp(v) for v in vals]
    s = sum(e)
    return [v / s for v in e]

def make_discrete(dominant: str):
    """Return plausible 7-emotion discrete distribution."""
    emotions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    dom_idx  = emotions.index(dominant)
    raw = [random.uniform(0.01, 0.15) for _ in emotions]
    raw[dom_idx] = random.uniform(0.55, 0.92)
    probs = softmax(raw)
    return dict(zip(emotions, [round(p, 10) for p in probs]))

def make_sentiment(dominant_emotion: str):
    """Return sentiment scores consistent with dominant emotion."""
    if dominant_emotion in ("joy", "surprise"):
        pos = random.uniform(0.75, 0.97)
        neg = random.uniform(0.005, 0.05)
        neu = max(0.0, 1.0 - pos - neg)
        label = "positive"
    elif dominant_emotion in ("anger", "disgust", "sadness", "fear"):
        neg = random.uniform(0.60, 0.90)
        pos = random.uniform(0.01, 0.15)
        neu = max(0.0, 1.0 - pos - neg)
        label = "negative"
    else:  # neutral
        neu = random.uniform(0.45, 0.70)
        pos = random.uniform(0.10, 0.30)
        neg = max(0.0, 1.0 - pos - neu)
        label = "positive" if pos > neg else "neutral"
    return {
        "label": label,
        "negative": round(neg, 15),
        "neutral":  round(neu, 15),
        "positive": round(pos, 15),
    }

def cyclical(value, period):
    angle = 2 * math.pi * value / period
    return round(math.sin(angle), 10), round(math.cos(angle), 10)

def get_time_of_day(hour):
    if 5  <= hour < 12: return "morning"
    if 12 <= hour < 17: return "afternoon"
    if 17 <= hour < 21: return "evening"
    return "night"

def get_season(month):
    if month in (12, 1, 2):  return "winter"
    if month in (3, 4, 5):   return "spring"
    if month in (6, 7, 8):   return "summer"
    return "autumn"

def recovery_day(captured_at: datetime, start: datetime):
    delta = (captured_at - start).total_seconds() / 86400
    return round(max(0.0, delta), 9)

# ── location templates ────────────────────────────────────────────────────────

LOCATIONS = [
    {
        "latitude": 61.2181, "longitude": -149.9003,
        "address": {"city": "Anchorage", "country": "United States", "road": "West 5th Avenue", "state": "Alaska"},
        "display_name": "USPS Mailbox, 939, West 5th Avenue, Anchorage, Alaska, 99501, United States",
        "place_type": "post_box"
    },
    {
        "latitude": 47.6062, "longitude": -122.3321,
        "address": {"city": "Seattle", "country": "United States", "road": "Pike Street", "state": "Washington"},
        "display_name": "Pike Place Market, 85, Pike Street, Seattle, Washington, 98101, United States",
        "place_type": "marketplace"
    },
    {
        "latitude": 40.7128, "longitude": -74.0060,
        "address": {"city": "New York City", "country": "United States", "road": "Broadway", "state": "New York"},
        "display_name": "Times Square, Broadway, New York City, New York, 10036, United States",
        "place_type": "attraction"
    },
    {
        "latitude": 37.7749, "longitude": -122.4194,
        "address": {"city": "San Francisco", "country": "United States", "road": "Market Street", "state": "California"},
        "display_name": "Civic Center, 1, Market Street, San Francisco, California, 94103, United States",
        "place_type": "public_building"
    },
    {
        "latitude": 51.5074, "longitude": -0.1278,
        "address": {"city": "London", "country": "United Kingdom", "road": "The Mall", "state": "England"},
        "display_name": "Buckingham Palace, The Mall, London, England, SW1A 1AA, United Kingdom",
        "place_type": "palace"
    },
    {
        "latitude": 19.0760, "longitude": 72.8777,
        "address": {"city": "Mumbai", "country": "India", "road": "Marine Drive", "state": "Maharashtra"},
        "display_name": "Marine Drive, Marine Drive, Mumbai, Maharashtra, 400020, India",
        "place_type": "road"
    },
    {
        "latitude": 28.6139, "longitude": 77.2090,
        "address": {"city": "New Delhi", "country": "India", "road": "Rajpath Avenue", "state": "Delhi"},
        "display_name": "India Gate, Rajpath Avenue, New Delhi, Delhi, 110001, India",
        "place_type": "monument"
    },
    {
        "latitude": 35.6762, "longitude": 139.6503,
        "address": {"city": "Tokyo", "country": "Japan", "road": "Shinjuku-dori", "state": "Tokyo"},
        "display_name": "Shinjuku Gyoen, Shinjuku-dori, Tokyo, 160-0014, Japan",
        "place_type": "park"
    },
    {
        "latitude": 48.8566, "longitude": 2.3522,
        "address": {"city": "Paris", "country": "France", "road": "Champs-Élysées", "state": "Île-de-France"},
        "display_name": "Arc de Triomphe, Champs-Élysées, Paris, Île-de-France, 75008, France",
        "place_type": "monument"
    },
    {
        "latitude": -33.8688, "longitude": 151.2093,
        "address": {"city": "Sydney", "country": "Australia", "road": "Circular Quay", "state": "New South Wales"},
        "display_name": "Sydney Opera House, Circular Quay, Sydney, New South Wales, 2000, Australia",
        "place_type": "theatre"
    },
]

# ── caption banks ─────────────────────────────────────────────────────────────

CAPTIONS_BY_EMOTION = {
    "joy":      [
        "felt some peace sitting by the plants today.",
        "best day out with friends, absolutely loved it!",
        "woke up feeling light and happy, rare but good.",
        "this sunset hit different today.",
        "cooked my favourite meal and it turned out perfect.",
        "small wins but they make me smile.",
    ],
    "neutral":  [
        "just another morning routine.",
        "quiet day indoors, nothing special.",
        "commute was uneventful today.",
        "got some reading done in the evening.",
        "grocery run done, productive enough.",
    ],
    "sadness":  [
        "missing home a little bit today.",
        "tired in a way sleep doesn't fix.",
        "some days just feel heavier than others.",
        "listening to old songs, feeling nostalgic.",
    ],
    "anger":    [
        "deadlines are killing me.",
        "stuck in traffic for an hour, not my day.",
        "another pointless meeting.",
        "code is broken and I have no idea why.",
    ],
    "fear":     [
        "big presentation tomorrow, nerves are real.",
        "waiting for test results, anxious.",
        "new city, don't know anyone yet.",
    ],
    "surprise": [
        "ran into an old friend completely by chance!",
        "did not expect that plot twist.",
        "got a surprise package in the mail.",
    ],
    "disgust":  [
        "that meal was a mistake.",
        "overheard a horrible conversation on the train.",
    ],
}

GEN_CAPTIONS = [
    "felt some peace sitting by the plants today.",
    "morning light through the window.",
    "city streets at dusk.",
    "quiet afternoon indoors.",
    "green spaces and fresh air.",
    "desk vibes and coffee.",
    "evening wind-down.",
    "weekend escape to nature.",
    "busy streets, busy mind.",
    "golden hour finally hit.",
    "food that brings comfort.",
    "simple moments, deep feelings.",
    "solitude that feels right.",
    "laughter with the best people.",
    "getting lost in a good book.",
]

CHIME_THEMES = {
    "joy":      ("Hope",         (0.70, 0.95)),
    "neutral":  ("Meaning",      (0.55, 0.80)),
    "sadness":  ("Identity",     (0.50, 0.75)),
    "anger":    ("Empowerment",  (0.60, 0.90)),
    "fear":     ("Empowerment",  (0.55, 0.85)),
    "surprise": ("Hope",         (0.65, 0.92)),
    "disgust":  ("Connectedness",(0.50, 0.75)),
}

IMAGE_PATH_TEMPLATE = (
    r"C:\Users\ANISH\OneDrive\Desktop\osC\DREAMS"
    r"\analysis_pipeline\data\processed\{memory_id}.jpg"
)

# ── main generator ────────────────────────────────────────────────────────────

def make_record(start_dt: datetime) -> dict:
    memory_id = hex_id(16)

    # timestamp
    offset_days  = random.uniform(0, 365)
    offset_hours = random.uniform(0, 24)
    captured_at  = start_dt + timedelta(days=offset_days, hours=offset_hours)
    # keep it in the past relative to "now" (2026-03-04)
    now_dt = datetime(2026, 3, 4, tzinfo=timezone.utc)
    if captured_at > now_dt:
        captured_at = now_dt - timedelta(days=random.uniform(0, 30))

    hour        = captured_at.hour
    month       = captured_at.month
    day_of_week = captured_at.weekday()   # 0=Mon … 6=Sun
    year        = captured_at.year

    # emotion
    dominant = random.choices(
        list(CAPTIONS_BY_EMOTION.keys()),
        weights=[4, 3, 2, 2, 1.5, 1.5, 0.5]
    )[0]
    discrete = make_discrete(dominant)

    # valence / arousal (nullable but we'll set them to floats here)
    valence  = None
    arousal  = None

    # sentiment
    sentiment = make_sentiment(dominant)

    # CHIME
    chime_cat, chime_range = CHIME_THEMES[dominant]
    chime_conf = random.uniform(*chime_range)

    # captions
    dom_captions = CAPTIONS_BY_EMOTION[dominant]
    gen_caption  = random.choice(GEN_CAPTIONS)
    has_user_caption = random.random() < 0.6
    if has_user_caption:
        caption_text   = random.choice(dom_captions)
        caption_source = "user"
    else:
        caption_text   = None
        caption_source = None

    # location
    loc = random.choice(LOCATIONS)

    # temporal cyclical
    sin_hour, cos_hour   = cyclical(hour, 24)
    sin_dow,  cos_dow    = cyclical(day_of_week, 7)
    sin_month, cos_month = cyclical(month - 1, 12)

    rec_day = recovery_day(captured_at, start_dt)
    time_of_day = get_time_of_day(hour)
    season      = get_season(month)

    captured_str = captured_at.strftime("%Y-%m-%dT%H:%M:%S.%f") + "+00:00"

    return {
        "caption":        caption_text,
        "caption_source": caption_source,
        "captured_at":    captured_str,
        "category":       None,
        "embeddings": {
            "caption": {"collection": "caption_embeddings", "dimensions": 384},
            "image":   {"collection": "image_embeddings",   "dimensions": 512},
        },
        "emotions": {
            "arousal":  arousal,
            "chime":    {"category": chime_cat, "confidence": round(chime_conf, 16)},
            "discrete": discrete,
            "dominant_emotion": dominant,
            "sentiment": sentiment,
            "valence":  valence,
        },
        "generated_caption": gen_caption,
        "image_path":        IMAGE_PATH_TEMPLATE.format(memory_id=memory_id),
        "is_duplicate":      False,
        "location": {
            "address":      loc["address"],
            "display_name": loc["display_name"],
            "latitude":     loc["latitude"],
            "longitude":    loc["longitude"],
            "place_type":   loc["place_type"],
        },
        "memory_id":         memory_id,
        "processing_status": "complete",
        "temporal": {
            "cyclical": {
                "cos_dow":   cos_dow,
                "cos_hour":  cos_hour,
                "cos_month": cos_month,
                "sin_dow":   sin_dow,
                "sin_hour":  sin_hour,
                "sin_month": sin_month,
            },
            "day_of_week":  day_of_week,
            "hour":         hour,
            "month":        month,
            "recovery_day": rec_day,
            "season":       season,
            "time_of_day":  time_of_day,
            "year":         year,
        },
        "user_id": "anishhhh",
    }


def main():
    output_path = r"C:\Users\ANISH\OneDrive\Desktop\osC\DREAMS\analysis_pipeline\data\raw\synthetic_dataset5.jsonl"
    n_records   = 300
    start_dt    = datetime(2025, 1, 1, tzinfo=timezone.utc)

    with open(output_path, "w", encoding="utf-8") as f:
        for _ in range(n_records):
            record = make_record(start_dt)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅  Written {n_records} records → {output_path}")


if __name__ == "__main__":
    main()
