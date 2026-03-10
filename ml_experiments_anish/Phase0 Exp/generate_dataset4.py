"""
generate_dataset4.py
──────────────────────────────────────────────────────────────────────────────
Generates synthetic_dataset4.jsonl — highly realistic Phase 1 experiment dataset.

KEY DESIGN PRINCIPLES:
  1. Orthogonal Image Centroids: Instead of random centroids (which collapse
     under PCA→UMAP), we use orthogonal basis vectors in 512D. This guarantees
     maximum angular separation between clusters.
  2. Controlled Noise: Intra-cluster noise std=0.20 (was 0.50). This keeps
     clusters tight enough to be separable but loose enough to be realistic.
  3. Maximally Distinct Emotions: Each cluster has a UNIQUE dominant emotion
     (anger, disgust, fear, joy, neutral, sadness) so the 15D emotion vector
     provides clear discriminative signal.
  4. Boundary Records: 8% of records are explicitly generated between specific
     cluster pairs (travel↔outdoors, home↔food, work↔fitness) to test edge cases.
  5. Imbalanced Clusters: Real-world long-tail distribution.
  6. 30% Missing Captions.
  7. Temporal Profiles per cluster.
  8. All records for single user 'anish'.
──────────────────────────────────────────────────────────────────────────────
Output: ../../analysis_pipeline/data/raw/synthetic_dataset4.jsonl
"""

import json
import math
import pathlib
import random
from datetime import datetime, timezone, timedelta
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────────
RANDOM_SEED   = 42
IMG_DIM       = 512
EMO_DIM       = 15
TOTAL_CORE    = 276        # core records (non-boundary)
BOUNDARY_PCT  = 0.08       # ~24 boundary records
INTRA_IMG_STD = 0.20       # tighter clusters
EMO_DISC_STD  = 0.08       # tighter emotion noise
EMO_VA_STD    = 0.10       # tighter valence/arousal noise

CHIME_IDX = {"Connectedness": 0, "Empowerment": 1, "Hope": 2, "Identity": 3, "Meaning": 4}
CHIME_LABELS = list(CHIME_IDX.keys())

# ── Cluster Definitions ────────────────────────────────────────────────────────
# CRITICAL: Each cluster gets a UNIQUE dominant emotion to maximize separation
# Cluster 0 (travel):   surprise-dominant
# Cluster 1 (home):     neutral-dominant
# Cluster 2 (work):     anger-dominant  (stress/frustration)
# Cluster 3 (outdoors): joy-dominant
# Cluster 4 (social):   disgust-dominant (awkwardness, overeating, hangovers)
# Cluster 5 (fitness):  fear-dominant   (challenge, pushing limits)

CLUSTERS_CONFIG = [
    {
        "id": 0, "theme": "travel_landmarks", "category": "travel", "count": 25,
        "locations": [("Tokyo", 35.68, 139.65), ("Paris", 48.85, 2.37), ("NYC", 40.71, -74.01), ("Rome", 41.89, 12.49)],
        "captions": [
            "Exploring {city} today.", "Landmark spotted in {city}!", "Travel vibes.", 
            "The architecture in {city} is amazing.", "Vacation mode: ON.", "Wish I could stay in {city} forever.",
            "Walking through history.", "Streets of {city}.", "A postcard moment.", "Soaking in the culture."
        ],
        "emo_profile": {
            "discrete_means": [0.03, 0.03, 0.05, 0.20, 0.09, 0.05, 0.55],  # SURPRISE dominant
            "val_arousal": (0.80, 0.75), "chime_weights": [0.15, 0.05, 0.45, 0.10, 0.25]
        },
        "temporal": {"days": [0,1,2,3,4,5,6], "hours": list(range(8, 22))}
    },
    {
        "id": 1, "theme": "home_daily", "category": "home", "count": 100,
        "locations": [("Home", 37.77, -122.42), ("Living Room", 37.77, -122.42), ("Kitchen", 37.77, -122.42)],
        "captions": [
            "Quiet morning.", "Home sweet home.", "Just chilling.", "Laundry day...", "Reading a book.",
            "New plant for the house!", "Cooking dinner.", "Cozy evening.", "Sunday reset.", "Desk setup.",
            "View from the window.", "Rainy day inside.", "Home office vibes.", "Organizing the shelf.", "Coffee break."
        ],
        "emo_profile": {
            "discrete_means": [0.05, 0.05, 0.03, 0.12, 0.60, 0.10, 0.05],  # NEUTRAL dominant
            "val_arousal": (0.50, 0.20), "chime_weights": [0.60, 0.05, 0.10, 0.15, 0.10]
        },
        "temporal": {"days": [0,1,2,3,4,5,6], "hours": [7,8,18,19,20,21,22]}
    },
    {
        "id": 2, "theme": "work_study", "category": "work", "count": 65,
        "locations": [("Office", 37.79, -122.40), ("Library", 37.79, -122.41), ("Coffee Shop", 37.78, -122.40)],
        "captions": [
            "Deep work.", "Meeting after meeting.", "Code is finally compiling.", "Study session.",
            "Working from {city}.", "Focus mode.", "Deadlines...", "Researching new ideas.",
            "Notebook full of sketches.", "Team lunch today.", "Late night at the office.", "Productive morning."
        ],
        "emo_profile": {
            "discrete_means": [0.50, 0.05, 0.15, 0.05, 0.15, 0.05, 0.05],  # ANGER dominant (stress)
            "val_arousal": (0.30, 0.70), "chime_weights": [0.10, 0.50, 0.10, 0.20, 0.10]
        },
        "temporal": {"days": [0,1,2,3,4], "hours": list(range(9, 18))}
    },
    {
        "id": 3, "theme": "outdoors_nature", "category": "outdoors", "count": 36,
        "locations": [("Park", 37.76, -122.48), ("Mountain", 37.92, -122.59), ("Beach", 37.83, -122.50)],
        "captions": [
            "Refreshing hike.", "Nature is healing.", "Cloudy day at the park.", "Mountain air.", 
            "Sunset hike.", "Trail running.", "Peace and quiet.", "Deep in the woods.", "Ocean breeze.",
            "Weekend escape.", "Finding balance."
        ],
        "emo_profile": {
            "discrete_means": [0.02, 0.03, 0.03, 0.60, 0.15, 0.07, 0.10],  # JOY dominant
            "val_arousal": (0.85, 0.35), "chime_weights": [0.10, 0.10, 0.15, 0.10, 0.55]
        },
        "temporal": {"days": [5,6], "hours": list(range(9, 17))}
    },
    {
        "id": 4, "theme": "food_social", "category": "social", "count": 35,
        "locations": [("Restaurant", 37.78, -122.41), ("Pub", 37.75, -122.42), ("Cafe", 37.79, -122.43)],
        "captions": [
            "Dinner with the crew.", "Best pizza in town!", "Drinks after work.", "Birthday dinner.",
            "Brunch time.", "Good food, better company.", "Happy hour.", "Trying this new spot.",
            "Celebrating!", "Table full of food."
        ],
        "emo_profile": {
            "discrete_means": [0.03, 0.50, 0.03, 0.15, 0.10, 0.09, 0.10],  # DISGUST dominant (overeating/hangovers)
            "val_arousal": (0.55, 0.65), "chime_weights": [0.50, 0.05, 0.25, 0.10, 0.10]
        },
        "temporal": {"days": [4,5,6], "hours": [19,20,21,22,23]}
    },
    {
        "id": 5, "theme": "fitness_health", "category": "fitness", "count": 15,
        "locations": [("Gym", 37.79, -122.39), ("Gym", 37.79, -122.39), ("Yoga Studio", 37.77, -122.43)],
        "captions": [
            "Morning workout.", "Leg day.", "Feeling strong.", "Sweat session.", "New PR!",
            "Morning yoga.", "Cardio is hardio.", "Post-gym glow.", "Consistency is key."
        ],
        "emo_profile": {
            "discrete_means": [0.05, 0.03, 0.55, 0.10, 0.07, 0.05, 0.15],  # FEAR dominant (challenge/limits)
            "val_arousal": (0.60, 0.90), "chime_weights": [0.05, 0.70, 0.10, 0.10, 0.05]
        },
        "temporal": {"days": [0,2,4], "hours": [6,7,8]}
    }
]

# Boundary pairs: records generated halfway between two cluster centroids
BOUNDARY_PAIRS = [
    (0, 3),  # travel ↔ outdoors  (both positive experiences in new places)
    (1, 4),  # home   ↔ social    (domestic vs social eating/gatherings)
    (2, 5),  # work   ↔ fitness   (similar indoor intensity)
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def cyclical_encode(value: float, period: float):
    angle = 2 * math.pi * value / period
    return round(math.sin(angle), 6), round(math.cos(angle), 6)

def make_orthogonal_centroids(n_clusters: int, dim: int, rng):
    """Create n_clusters unit vectors that are mutually orthogonal in `dim` dimensions.
    
    This guarantees maximum angular separation between cluster centres,
    preventing the collapse that happens with random centroids under PCA→UMAP.
    """
    # QR decomposition of a random matrix gives orthonormal columns
    A = rng.standard_normal((dim, n_clusters))
    Q, _ = np.linalg.qr(A)
    # Each column of Q is a unit vector, all mutually orthogonal
    centroids = {}
    for i in range(n_clusters):
        centroids[i] = Q[:, i]
    return centroids

def generate_emotion_vector(emo_profile, rng):
    """Generate a single 15D emotion vector from a cluster's emotion profile."""
    p = emo_profile
    
    # Discrete emotions with controlled noise
    disc_raw = rng.normal(p["discrete_means"], EMO_DISC_STD)
    disc_raw = np.clip(disc_raw, 0.01, 1.0)  # floor at 0.01 to avoid zeros
    disc_raw /= disc_raw.sum()  # normalize to probability simplex
    
    ekman_keys = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    dominant_emotion = ekman_keys[int(np.argmax(disc_raw))]
    
    valence = float(np.clip(rng.normal(p["val_arousal"][0], EMO_VA_STD), 0, 1))
    arousal = float(np.clip(rng.normal(p["val_arousal"][1], EMO_VA_STD), 0, 1))
    
    # CHIME
    chime_probs = np.array(p["chime_weights"], dtype=float)
    chime_probs += rng.dirichlet(np.ones(5) * 0.5) * 0.15  # mild noise
    chime_probs /= chime_probs.sum()
    chime_cat = CHIME_LABELS[int(np.argmax(chime_probs))]
    chime_conf = float(np.clip(rng.normal(0.75, 0.10), 0.35, 0.95))
    
    emotions_doc = {
        "discrete": {k: round(float(v), 6) for k, v in zip(ekman_keys, disc_raw)},
        "dominant_emotion": dominant_emotion,
        "valence": round(valence, 6),
        "arousal": round(arousal, 6),
        "sentiment": {"label": "positive" if valence > 0.55 else ("negative" if valence < 0.45 else "neutral")},
        "chime": {"category": chime_cat, "confidence": round(chime_conf, 6)}
    }
    
    chime_onehot = [0.0] * 5
    chime_onehot[CHIME_IDX[chime_cat]] = 1.0
    emo_vec = ([round(float(v), 6) for v in disc_raw] 
               + [round(valence, 6), round(arousal, 6)] 
               + chime_onehot 
               + [round(chime_conf, 6)])
    
    return emo_vec, emotions_doc, dominant_emotion

def generate_dataset(out_path: pathlib.Path):
    rng = np.random.default_rng(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # 1. Orthogonal Image Centroids
    n_clusters = len(CLUSTERS_CONFIG)
    centroids = make_orthogonal_centroids(n_clusters, IMG_DIM, rng)
    
    # Verify orthogonality
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            dot = np.dot(centroids[i], centroids[j])
            assert abs(dot) < 1e-10, f"Centroids {i} and {j} not orthogonal: dot={dot}"
    print(f"✓ {n_clusters} orthogonal centroids created in {IMG_DIM}D (all pairwise dot products ≈ 0)")

    all_records = []
    record_counter = 0

    # ── Core records ───────────────────────────────────────────────────────────
    for cluster in CLUSTERS_CONFIG:
        cid = cluster["id"]
        centroid = centroids[cid]
        
        for i in range(cluster["count"]):
            record_counter += 1
            
            # --- Image Embedding (orthogonal centroid + controlled noise → L2 norm) ---
            noise = rng.normal(0, INTRA_IMG_STD, size=IMG_DIM)
            img_vec = centroid + noise
            img_vec = img_vec / np.linalg.norm(img_vec)
            img_vec = [round(float(x), 6) for x in img_vec]

            # --- Emotion Vector ---
            emo_vec, emotions_doc, dominant_emotion = generate_emotion_vector(cluster["emo_profile"], rng)

            # --- Caption (30% null) ---
            has_caption = rng.random() > 0.3
            loc_choice = random.choice(cluster["locations"])
            city_name = loc_choice[0]
            cap_text = random.choice(cluster["captions"]).format(city=city_name) if has_caption else None

            # --- Temporal ---
            base_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
            found_time = False
            while not found_time:
                days_offset = rng.integers(0, 400)
                target_date = base_date + timedelta(days=int(days_offset))
                if target_date.weekday() in cluster["temporal"]["days"]:
                    hour = random.choice(cluster["temporal"]["hours"])
                    minute = rng.integers(0, 59)
                    target_date = target_date.replace(hour=hour, minute=int(minute))
                    found_time = True
            
            sh, ch = cyclical_encode(target_date.hour, 24)
            sd, cd = cyclical_encode(target_date.weekday(), 7)
            sm, cm = cyclical_encode(target_date.month, 12)
            temporal = {
                "hour": target_date.hour, "day_of_week": target_date.weekday(), "month": target_date.month,
                "cyclical": {"sin_hour": sh, "cos_hour": ch, "sin_dow": sd, "cos_dow": cd, "sin_month": sm, "cos_month": cm}
            }

            memory_id = f"synth4_{cid}_{i:03d}"
            
            all_records.append({
                "memory_id": memory_id,
                "user_id": "anish",
                "category": cluster["category"],
                "caption": cap_text,
                "generated_caption": random.choice(cluster["captions"]).format(city=city_name),
                "captured_at": target_date.isoformat(),
                "emotions": emotions_doc,
                "location": {"latitude": loc_choice[1], "longitude": loc_choice[2]},
                "temporal": temporal,
                "embeddings": {
                    "image": {"vector": img_vec, "dimensions": IMG_DIM},
                    "emotion": {
                        "vector": emo_vec, 
                        "dimensions": EMO_DIM,
                        "schema": "anger,disgust,fear,joy,neutral,sadness,surprise,valence,arousal,connectedness,empowerment,hope,identity,meaning,confidence"
                    }
                },
                "synthetic": {"cluster_id": cid, "theme": cluster["theme"], "is_boundary": False}
            })

    # ── Boundary records (8%) ──────────────────────────────────────────────────
    n_boundary = int(len(all_records) * BOUNDARY_PCT)
    boundary_per_pair = max(1, n_boundary // len(BOUNDARY_PAIRS))
    
    print(f"✓ Generating {boundary_per_pair * len(BOUNDARY_PAIRS)} boundary records across {len(BOUNDARY_PAIRS)} pairs")
    
    for cid_a, cid_b in BOUNDARY_PAIRS:
        cluster_a = CLUSTERS_CONFIG[cid_a]
        cluster_b = CLUSTERS_CONFIG[cid_b]
        centroid_a = centroids[cid_a]
        centroid_b = centroids[cid_b]
        
        for i in range(boundary_per_pair):
            record_counter += 1
            
            # Image: interpolate between the two centroids with noise
            mix = rng.uniform(0.35, 0.65)  # near the midpoint
            img_vec = mix * centroid_a + (1 - mix) * centroid_b
            img_vec += rng.normal(0, INTRA_IMG_STD * 0.5, size=IMG_DIM)
            img_vec = img_vec / np.linalg.norm(img_vec)
            img_vec = [round(float(x), 6) for x in img_vec]
            
            # Emotion: randomly pick one of the two cluster profiles
            chosen_cluster = random.choice([cluster_a, cluster_b])
            emo_vec, emotions_doc, dominant_emotion = generate_emotion_vector(chosen_cluster["emo_profile"], rng)
            
            # Caption
            has_caption = rng.random() > 0.3
            all_caps = cluster_a["captions"] + cluster_b["captions"]
            all_locs = cluster_a["locations"] + cluster_b["locations"]
            loc_choice = random.choice(all_locs)
            cap_text = random.choice(all_caps).format(city=loc_choice[0]) if has_caption else None
            
            # Temporal (merge temporal windows)
            all_days = list(set(cluster_a["temporal"]["days"] + cluster_b["temporal"]["days"]))
            all_hours = list(set(cluster_a["temporal"]["hours"] + cluster_b["temporal"]["hours"]))
            base_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
            found_time = False
            while not found_time:
                days_offset = rng.integers(0, 400)
                target_date = base_date + timedelta(days=int(days_offset))
                if target_date.weekday() in all_days:
                    hour = random.choice(all_hours)
                    minute = rng.integers(0, 59)
                    target_date = target_date.replace(hour=hour, minute=int(minute))
                    found_time = True
            
            sh, ch = cyclical_encode(target_date.hour, 24)
            sd, cd = cyclical_encode(target_date.weekday(), 7)
            sm, cm = cyclical_encode(target_date.month, 12)
            temporal = {
                "hour": target_date.hour, "day_of_week": target_date.weekday(), "month": target_date.month,
                "cyclical": {"sin_hour": sh, "cos_hour": ch, "sin_dow": sd, "cos_dow": cd, "sin_month": sm, "cos_month": cm}
            }
            
            # Assign ground truth to the closer cluster
            memory_id = f"synth4_bnd_{cid_a}{cid_b}_{i:03d}"
            gt_cluster = cid_a if mix > 0.5 else cid_b
            gt_theme = CLUSTERS_CONFIG[gt_cluster]["theme"]
            
            all_records.append({
                "memory_id": memory_id,
                "user_id": "anish",
                "category": chosen_cluster["category"],
                "caption": cap_text,
                "generated_caption": random.choice(all_caps).format(city=loc_choice[0]),
                "captured_at": target_date.isoformat(),
                "emotions": emotions_doc,
                "location": {"latitude": loc_choice[1], "longitude": loc_choice[2]},
                "temporal": temporal,
                "embeddings": {
                    "image": {"vector": img_vec, "dimensions": IMG_DIM},
                    "emotion": {
                        "vector": emo_vec,
                        "dimensions": EMO_DIM,
                        "schema": "anger,disgust,fear,joy,neutral,sadness,surprise,valence,arousal,connectedness,empowerment,hope,identity,meaning,confidence"
                    }
                },
                "synthetic": {"cluster_id": gt_cluster, "theme": gt_theme, "is_boundary": True}
            })

    random.shuffle(all_records)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  Generated {len(all_records)} records to {out_path}")
    print(f"  Core records: {sum(c['count'] for c in CLUSTERS_CONFIG)}")
    print(f"  Boundary records: {boundary_per_pair * len(BOUNDARY_PAIRS)}")
    print(f"  Cluster distribution:")
    for c in CLUSTERS_CONFIG:
        print(f"    [{c['id']}] {c['theme']:20s} → {c['count']:3d} core  (dominant emo: {['anger','disgust','fear','joy','neutral','sadness','surprise'][np.argmax(c['emo_profile']['discrete_means'])]})")
    print(f"{'='*60}")


if __name__ == "__main__":
    out = pathlib.Path(r"C:\Users\ANISH\OneDrive\Desktop\osC\DREAMS\analysis_pipeline\data\raw\synthetic_dataset4.jsonl")
    generate_dataset(out)
