"""
generate_dataset3.py
──────────────────────────────────────────────────────────────────────────────
Generates synthetic_dataset3.jsonl — realistic Phase 1 experiment dataset.

Key differences from synthetic_dataset2.jsonl:
  1. Emotion vector replaces caption embedding
     - 15D: [7 discrete Ekman] + [valence] + [arousal] + [5 CHIME one-hot] + [CHIME confidence]
     - Stored as embeddings.emotion.vector (dimensions: 15)
  0. Image embedding dimensions updated to 512D (real CLIP ViT output)
     Previously was placeholder 64D — now matches actual pipeline output
  2. Harder, more realistic cluster structure
     - Inter-cluster distance reduced (blobs overlap at boundaries)
     - Higher intra-cluster variance (std 0.9 vs clean 0.4)
     - 12% cross-cluster contamination (records near two cluster centroids)
     - CHIME + dominant emotion profiles not perfectly uniform per cluster
     - Some within-cluster emotional diversity (e.g. some travel memories are sad)
  3. No embeddings.caption field
──────────────────────────────────────────────────────────────────────────────
Run:  python generate_dataset3.py
Output: ../../analysis_pipeline/data/raw/synthetic_dataset3.jsonl
"""

import json
import math
import pathlib
import random
import uuid
from datetime import datetime, timezone, timedelta

import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────────
RANDOM_SEED   = 7
N_PER_CLUSTER = 40          # 6 × 40 = 240 records total
IMG_DIM       = 512         # matches real pipeline (CLIP ViT-B/32 → 512D)
EMO_DIM       = 15          # emotion vector dimensionality

# CHIME 5-class index map (from production_chime_model config.json)
CHIME_IDX = {"Connectedness": 0, "Empowerment": 1, "Hope": 2, "Identity": 3, "Meaning": 4}
CHIME_LABELS = list(CHIME_IDX.keys())

# ── Cluster definitions ────────────────────────────────────────────────────────
# Each cluster has:
#   theme, category, locations, captions, and TARGET emotion/embedding profiles
# "Realism" is achieved by:
#   - Sharing similar target valence/arousal between neighbouring clusters
#   - Emotion noise std high enough to blur boundaries
#   - Image centroid separation smaller than dataset2

CLUSTERS = [
    # ─ Cluster 0: travel_landmarks ─────────────────────────────────────────────
    {
        "cluster_id":  0,
        "theme":       "travel_landmarks",
        "category":    "travel",
        "locations": [
            ("Tokyo",    35.68,  139.65, "Tokyo, Japan"),
            ("Paris",    48.85,    2.37, "Paris, France"),
            ("New York", 40.71,  -74.01, "New York, United States"),
        ],
        "captions": [
            "Travel snapshot from {city} — feeling inspired.",
            "A bright day exploring {city} streets.",
            "Walking near a famous landmark in {city}.",
        ],
        # Emotion profile: high joy, high valence, medium-high arousal
        # CHIME: mostly Hope & Meaning (but some Connectedness bleeds in)
        "emo_profile": {
            "discrete_means":  [0.04, 0.04, 0.05, 0.70, 0.06, 0.05, 0.06],  # joy dominant
            "discrete_std":    0.08,
            "valence_mean":    0.72, "valence_std":  0.12,
            "arousal_mean":    0.60, "arousal_std":  0.12,
            "chime_weights":   [0.10, 0.05, 0.55, 0.10, 0.20],  # mostly Hope + Meaning
            "conf_mean":       0.88, "conf_std":     0.05,
        },
        # Image centroid in 64D (will be seeded)
        "img_centroid_seed": 101,
    },
    # ─ Cluster 1: home_daily ────────────────────────────────────────────────────
    {
        "cluster_id":  1,
        "theme":       "home_daily",
        "category":    "home",
        "locations": [
            ("Home",     37.77, -122.42, "San Francisco, United States"),
            ("Home",     51.51,   -0.13, "London, United Kingdom"),
            ("Home",     48.85,    2.35, "Paris, France"),
        ],
        "captions": [
            "Quiet morning at home in {city}.",
            "Relaxing evening, feeling settled.",
            "Daily life snapshot — {city}.",
        ],
        # Emotion profile: neutral/mild joy, medium valence, LOW arousal
        # Realistic blur: some records look like food_social (shared Connectedness)
        "emo_profile": {
            "discrete_means":  [0.04, 0.03, 0.04, 0.38, 0.32, 0.10, 0.09],  # neutral+joy
            "discrete_std":    0.09,
            "valence_mean":    0.52, "valence_std":  0.14,
            "arousal_mean":    0.28, "arousal_std":  0.11,
            "chime_weights":   [0.55, 0.10, 0.15, 0.12, 0.08],  # mostly Connectedness
            "conf_mean":       0.82, "conf_std":     0.07,
        },
        "img_centroid_seed": 202,
    },
    # ─ Cluster 2: work_study ────────────────────────────────────────────────────
    {
        "cluster_id":  2,
        "theme":       "work_study",
        "category":    "work",
        "locations": [
            ("Office",   37.79, -122.40, "San Francisco, United States"),
            ("Library",  51.50,   -0.12, "London, United Kingdom"),
            ("Campus",   40.73,  -73.99, "New York, United States"),
        ],
        "captions": [
            "Deep in work today — {city} office.",
            "Study session, focused and determined.",
            "Another productive day at the {city} campus.",
        ],
        # Emotion profile: neutral dominant, medium valence, medium-high arousal
        # Realistic blur: bleeds into fitness_health (shared Empowerment + medium arousal)
        "emo_profile": {
            "discrete_means":  [0.07, 0.03, 0.08, 0.25, 0.38, 0.12, 0.07],  # neutral dominant
            "discrete_std":    0.09,
            "valence_mean":    0.44, "valence_std":  0.15,
            "arousal_mean":    0.52, "arousal_std":  0.14,
            "chime_weights":   [0.08, 0.42, 0.08, 0.30, 0.12],  # Empowerment + Identity
            "conf_mean":       0.79, "conf_std":     0.08,
        },
        "img_centroid_seed": 303,
    },
    # ─ Cluster 3: outdoors_nature ───────────────────────────────────────────────
    {
        "cluster_id":  3,
        "theme":       "outdoors_nature",
        "category":    "outdoors",
        "locations": [
            ("Yosemite", 37.75, -119.59, "Yosemite, United States"),
            ("Alps",     46.85,    7.20, "Bern, Switzerland"),
            ("Kyoto",    35.01,  135.77, "Kyoto, Japan"),
        ],
        "captions": [
            "Breathtaking views near {city}.",
            "A peaceful hike in the {city} wilderness.",
            "Nature photography session — {city} landscape.",
        ],
        # Emotion profile: joy + surprise mix, HIGH valence, MEDIUM arousal
        # Realistic blur: bleeds into travel_landmarks (both high joy/valence/Hope)
        "emo_profile": {
            "discrete_means":  [0.03, 0.03, 0.06, 0.58, 0.08, 0.05, 0.17],  # joy+surprise
            "discrete_std":    0.09,
            "valence_mean":    0.76, "valence_std":  0.11,
            "arousal_mean":    0.50, "arousal_std":  0.13,
            "chime_weights":   [0.08, 0.05, 0.38, 0.05, 0.44],  # Meaning + Hope
            "conf_mean":       0.85, "conf_std":     0.06,
        },
        "img_centroid_seed": 404,
    },
    # ─ Cluster 4: food_social ───────────────────────────────────────────────────
    {
        "cluster_id":  4,
        "theme":       "food_social",
        "category":    "social",
        "locations": [
            ("Tokyo",    35.69,  139.70, "Tokyo, Japan"),
            ("London",   51.51,   -0.09, "London, United Kingdom"),
            ("New York", 40.73,  -73.98, "New York, United States"),
        ],
        "captions": [
            "Amazing meal with friends in {city}.",
            "Food and laughter — {city} dining.",
            "Sharing a great meal at a {city} restaurant.",
        ],
        # Emotion profile: joy + high valence + HIGH arousal
        # Realistic blur: bleeds into home_daily (shared Connectedness)
        "emo_profile": {
            "discrete_means":  [0.03, 0.03, 0.04, 0.68, 0.06, 0.05, 0.11],  # joy dominant
            "discrete_std":    0.08,
            "valence_mean":    0.78, "valence_std":  0.10,
            "arousal_mean":    0.72, "arousal_std":  0.11,
            "chime_weights":   [0.40, 0.05, 0.35, 0.10, 0.10],  # Connectedness + Hope
            "conf_mean":       0.87, "conf_std":     0.05,
        },
        "img_centroid_seed": 505,
    },
    # ─ Cluster 5: fitness_health ────────────────────────────────────────────────
    {
        "cluster_id":  5,
        "theme":       "fitness_health",
        "category":    "fitness",
        "locations": [
            ("Gym",      37.80, -122.41, "San Francisco, United States"),
            ("Park",     51.48,   -0.17, "London, United Kingdom"),
            ("Track",    40.75,  -73.97, "New York, United States"),
        ],
        "captions": [
            "Morning run complete — feeling strong in {city}.",
            "Gym session done. {city} energy.",
            "Pushing limits at the {city} track.",
        ],
        # Emotion profile: mixed (fear/challenge), medium valence, HIGH arousal
        # Realistic blur: bleeds into work_study (shared Empowerment + medium val)
        "emo_profile": {
            "discrete_means":  [0.05, 0.03, 0.14, 0.38, 0.20, 0.10, 0.10],  # mixed
            "discrete_std":    0.10,
            "valence_mean":    0.56, "valence_std":  0.15,
            "arousal_mean":    0.78, "arousal_std":  0.11,
            "chime_weights":   [0.08, 0.55, 0.15, 0.15, 0.07],  # Empowerment dominant
            "conf_mean":       0.81, "conf_std":     0.07,
        },
        "img_centroid_seed": 606,
    },
]

# ── Boundary pairs (which cluster pairs should share contamination) ───────────
# (cluster_a, cluster_b, fraction_of_N_to_contaminate_from_a, vice_versa)
BOUNDARY_PAIRS = [
    (0, 3),   # travel_landmarks ↔ outdoors_nature (both high joy/valence/Hope)
    (1, 4),   # home_daily ↔ food_social (shared Connectedness)
    (2, 5),   # work_study ↔ fitness_health (shared Empowerment + arousal)
]
CONTAMINATION_RATE = 0.10   # 10% of records per cluster get boundary noise

# ── Image embedding centroid generation ───────────────────────────────────────
# Centroids placed on a sphere with controlled inter-centroid distance
# INTRA_STD controls cluster spread; smaller = more overlap
INTRA_STD          = 0.90   # intra-cluster noise std (was ~0.4 in dataset2)
CENTROID_SCALE     = 2.0    # centroid magnitude; smaller = more overlap between clusters


def make_img_centroid(seed: int, dim: int = IMG_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    v = v / np.linalg.norm(v)
    return v * CENTROID_SCALE


def sample_img_embedding(centroid: np.ndarray, rng: np.random.Generator) -> list:
    noise = rng.normal(0, INTRA_STD, size=centroid.shape)
    vec   = centroid + noise
    return [round(float(x), 6) for x in vec]


# ── Emotion vector generation ──────────────────────────────────────────────────

def sample_emotion_record(profile: dict, rng: np.random.Generator):
    """
    Returns:
        emotions: dict  (full emotion sub-document, matches existing schema)
        emo_vec: list   (15D embedding vector)
    """
    # --- discrete Ekman (7D) ---
    discrete_raw = rng.normal(profile["discrete_means"], profile["discrete_std"])
    discrete_raw = np.clip(discrete_raw, 0.0, 1.0)
    discrete_raw = discrete_raw / (discrete_raw.sum() + 1e-9)  # normalize to sum=1

    ekman_keys = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    discrete_dict = {k: round(float(v), 6) for k, v in zip(ekman_keys, discrete_raw)}
    dominant_emotion = ekman_keys[int(np.argmax(discrete_raw))]

    # --- valence & arousal ---
    valence = float(np.clip(rng.normal(profile["valence_mean"], profile["valence_std"]), 0.0, 1.0))
    arousal = float(np.clip(rng.normal(profile["arousal_mean"], profile["arousal_std"]), 0.0, 1.0))

    # --- sentiment (derived from valence) ---
    pos = float(np.clip(rng.normal(valence + 0.10, 0.08), 0.01, 0.99))
    neg = float(np.clip(rng.normal(1.0 - valence + 0.05, 0.08), 0.01, 0.99))
    tot = pos + neg + 0.05
    pos_n, neg_n, neu_n = pos/tot, neg/tot, 0.05/tot
    sentiment_label = "positive" if pos_n > neg_n else ("negative" if neg_n > 0.4 else "neutral")

    # --- CHIME ---
    chime_probs = np.array(profile["chime_weights"], dtype=np.float64)
    chime_probs += rng.dirichlet(np.ones(5) * 0.5) * 0.25  # add Dirichlet noise
    chime_probs /= chime_probs.sum()
    chime_class = CHIME_LABELS[int(np.argmax(chime_probs))]
    chime_conf  = float(np.clip(rng.normal(profile["conf_mean"], profile["conf_std"]), 0.60, 0.99))

    emotions = {
        "discrete": discrete_dict,
        "dominant_emotion": dominant_emotion,
        "valence": round(valence, 6),
        "arousal": round(arousal, 6),
        "sentiment": {
            "label": sentiment_label,
            "positive": round(pos_n, 6),
            "negative": round(neg_n, 6),
            "neutral":  round(neu_n, 6),
        },
        "chime": {
            "category":   chime_class,
            "confidence": round(chime_conf, 6),
        },
    }

    # --- 15D emotion vector ---
    chime_onehot = [0.0] * 5
    chime_onehot[CHIME_IDX[chime_class]] = 1.0

    emo_vec = (
        [round(float(discrete_raw[i]), 6) for i in range(7)]   # 7D discrete
        + [round(valence, 6), round(arousal, 6)]                # 2D circumplex
        + [round(float(x), 6) for x in chime_onehot]           # 5D CHIME one-hot
        + [round(chime_conf, 6)]                                # 1D confidence
    )  # total 15D
    assert len(emo_vec) == EMO_DIM

    return emotions, emo_vec


# ── Temporal helpers ───────────────────────────────────────────────────────────

def cyclical_encode(value: float, period: float):
    angle = 2 * math.pi * value / period
    return round(math.sin(angle), 6), round(math.cos(angle), 6)

def sample_temporal(rng: np.random.Generator):
    base    = datetime(2025, 1, 1, tzinfo=timezone.utc)
    offset  = timedelta(days=int(rng.integers(0, 450)), hours=int(rng.integers(6, 22)))
    dt      = base + offset
    hour    = dt.hour
    dow     = dt.weekday()
    month   = dt.month

    season_map = {12: "winter", 1: "winter", 2: "winter",
                  3: "spring", 4: "spring", 5: "spring",
                  6: "summer", 7: "summer", 8: "summer",
                  9: "fall",   10: "fall",  11: "fall"}
    tod_map = {range(0,6): "night", range(6,12): "morning",
               range(12,17): "afternoon", range(17,21): "evening"}
    time_of_day = "night"
    for r, t in tod_map.items():
        if hour in r:
            time_of_day = t; break

    sin_h,   cos_h   = cyclical_encode(hour,  24)
    sin_dow, cos_dow = cyclical_encode(dow,    7)
    sin_m,   cos_m   = cyclical_encode(month, 12)

    return {
        "hour":         hour,
        "day_of_week":  dow,
        "month":        month,
        "year":         dt.year,
        "season":       season_map[month],
        "time_of_day":  time_of_day,
        "recovery_day": round(float(offset.days + hour / 24.0), 1),
        "cyclical": {
            "sin_hour": sin_h, "cos_hour": cos_h,
            "sin_dow":  sin_dow, "cos_dow": cos_dow,
            "sin_month": sin_m, "cos_month": cos_m,
        },
    }, dt.isoformat()


# ── Main generation ────────────────────────────────────────────────────────────

def generate_dataset(out_path: pathlib.Path):
    rng = np.random.default_rng(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    users = ["anish", "researcher01", "demo_user", "dev_user"]

    # Pre-generate image centroids
    centroids = {c["cluster_id"]: make_img_centroid(c["img_centroid_seed"]) for c in CLUSTERS}

    # Pre-compute boundary centroids (midpoint between two cluster centroids)
    boundary_centroids = {}
    for ca, cb in BOUNDARY_PAIRS:
        mid = (centroids[ca] + centroids[cb]) / 2.0
        boundary_centroids[(ca, cb)] = mid

    all_records = []
    record_counter = 0

    for cluster in CLUSTERS:
        cid      = cluster["cluster_id"]
        theme    = cluster["theme"]
        category = cluster["category"]
        centroid = centroids[cid]

        n_boundary = int(N_PER_CLUSTER * CONTAMINATION_RATE)
        n_core     = N_PER_CLUSTER - n_boundary

        for i in range(N_PER_CLUSTER):
            record_counter += 1
            is_boundary = i < n_boundary

            # Image embedding: core = from own centroid; boundary = from midpoint centroid
            if is_boundary:
                # Find a matching boundary pair for this cluster
                bc_key = next(
                    (k for k in boundary_centroids if cid in k),
                    None
                )
                bc = boundary_centroids[bc_key] if bc_key else centroid
                img_vec = sample_img_embedding(bc, rng)
            else:
                img_vec = sample_img_embedding(centroid, rng)

            # Emotion: always from cluster's own profile (real memories feel their theme)
            emotions, emo_vec = sample_emotion_record(cluster["emo_profile"], rng)

            # Location: pick from cluster's location list
            loc_template = cluster["locations"][i % len(cluster["locations"])]
            city, lat, lon, display_name = loc_template
            country_parts = display_name.split(", ")

            # Caption
            cap_template = cluster["captions"][i % len(cluster["captions"])]
            has_user_caption = rng.random() > 0.15  # 85% have user caption
            caption_text = cap_template.format(city=city)

            # Temporal
            temporal, captured_at = sample_temporal(rng)

            memory_id = f"synth3_{cid}_{i:03d}_{record_counter:04d}"

            record = {
                "memory_id": memory_id,
                "user_id":   random.choice(users),
                "image_path": f"analysis_pipeline/data/processed/{memory_id}.jpg",
                "caption":    caption_text if has_user_caption else None,
                "generated_caption": caption_text,
                "caption_source": "user" if has_user_caption else "generated",
                "category": category,
                "captured_at": captured_at,
                "is_duplicate": False,
                "emotions": emotions,
                "location": {
                    "latitude":     round(lat + float(rng.normal(0, 0.01)), 6),
                    "longitude":    round(lon + float(rng.normal(0, 0.01)), 6),
                    "display_name": display_name,
                    "place_type":   "synthetic",
                    "address": {
                        "road":    None,
                        "city":    country_parts[0] if len(country_parts) > 0 else None,
                        "state":   country_parts[-2] if len(country_parts) > 2 else None,
                        "country": country_parts[-1] if len(country_parts) > 1 else None,
                    },
                },
                "temporal": temporal,
                "embeddings": {
                    "image": {
                        "vector": img_vec,
                        "dimensions": IMG_DIM,
                    },
                    "emotion": {
                        "vector": emo_vec,
                        "dimensions": EMO_DIM,
                        "schema": (
                            "anger,disgust,fear,joy,neutral,sadness,surprise,"
                            "valence,arousal,"
                            "chime_connectedness,chime_empowerment,chime_hope,"
                            "chime_identity,chime_meaning,chime_confidence"
                        ),
                    },
                },
                "processing_status": "complete",
                "synthetic": {
                    "cluster_id": cid,
                    "theme":      theme,
                    "is_boundary_record": is_boundary,
                },
            }
            all_records.append(record)

    # Shuffle so clusters are not in order (realistic)
    random.shuffle(all_records)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Written {len(all_records)} records → {out_path}")

    # ── Validation report ──────────────────────────────────────────────────────
    from collections import Counter
    cluster_counts = Counter(r["synthetic"]["cluster_id"] for r in all_records)
    boundary_counts = Counter(
        r["synthetic"]["cluster_id"]
        for r in all_records
        if r["synthetic"]["is_boundary_record"]
    )
    print("\nCluster distribution:")
    for cid in sorted(cluster_counts):
        theme = [c["theme"] for c in CLUSTERS if c["cluster_id"] == cid][0]
        print(f"  Cluster {cid} ({theme:22s}): {cluster_counts[cid]:3d} total, "
              f"{boundary_counts[cid]:2d} boundary ({100*boundary_counts[cid]/cluster_counts[cid]:.0f}%)")

    # Check emotion vector dimensions
    all_dims = [r["embeddings"]["emotion"]["dimensions"] for r in all_records]
    all_lens = [len(r["embeddings"]["emotion"]["vector"]) for r in all_records]
    print(f"\nEmotion vector: all dims={set(all_dims)}, all lengths={set(all_lens)}")

    img_dims = [len(r["embeddings"]["image"]["vector"]) for r in all_records]
    print(f"Image vector:   all lengths={set(img_dims)}")

    print("\nSchema keys present in first record:")
    first = all_records[0]
    print(f"  embeddings.image:    {list(first['embeddings']['image'].keys())}")
    print(f"  embeddings.emotion:  {list(first['embeddings']['emotion'].keys())}")
    assert "caption" not in first["embeddings"], "FAIL: caption embedding should not exist"
    print("  (no embeddings.caption field — correct)")

    # Check CHIME categories
    chime_dist = Counter(r["emotions"]["chime"]["category"] for r in all_records)
    print(f"\nCHIME distribution: {dict(chime_dist)}")

    # Check dominant emotion distribution
    emo_dist = Counter(r["emotions"]["dominant_emotion"] for r in all_records)
    print(f"Dominant emotion:   {dict(emo_dist)}")

    return all_records


if __name__ == "__main__":
    repo_root  = pathlib.Path(__file__).parent.parent.parent.resolve()
    out_path   = repo_root / "analysis_pipeline" / "data" / "raw" / "synthetic_dataset3.jsonl"
    print(f"Generating synthetic_dataset3.jsonl → {out_path}")
    generate_dataset(out_path)
    print("\nDone.")
