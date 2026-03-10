# Narrative Structure Analysis

End-to-end documentation for the graph-based emotional narrative analysis feature in DREAMS.

---

## Table of Contents

1. [What This Feature Does](#what-this-feature-does)
2. [Pipeline Overview](#pipeline-overview)
3. [Data Flow](#data-flow)
4. [Analytics Module — `graph_analysis.py`](#analytics-module)
5. [API Endpoint](#api-endpoint)
6. [Dashboard Visualization](#dashboard-visualization)
7. [Configuration](#configuration)
8. [File Map](#file-map)
9. [Testing](#testing)
10. [Known Limitations](#known-limitations)
11. [Future Roadmap](#future-roadmap)

---

## What This Feature Does

When users upload posts (caption + image + timestamp), the platform runs sentiment analysis and stores the result. Over time, this builds up a chronological stream of emotional observations for each user.

This feature takes that stream and:

1. **Groups** posts into **episodes** (clusters of emotionally related posts that are close in time)
2. **Connects** episodes into a **directed graph** based on temporal proximity
3. **Analyzes** the graph to compute structural metrics — centrality, transitions, cycles, density
4. **Exposes** all of this via a JSON API
5. **Visualizes** it on an interactive dashboard page

The result: a researcher can look at a user's emotional narrative structure, see which emotional states dominate, which states act as "bridges", and which emotional loops the user tends to get stuck in.

---

## Pipeline Overview

```
MongoDB posts (user_id, caption, timestamp, sentiment)
        │
        ▼
┌─────────────────────┐
│  EmotionTimeline     │   Chronologically sorted list of EmotionEvent objects
│  (builder.py)        │   Each event = one post with timestamp + sentiment label
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────┐
│  Episodes                │   Groups of events clustered by time gap
│  (episode_segmentation)  │   Posts within 24h of each other → same episode
└─────────┬───────────────┘
          │
          ▼
┌──────────────────────────────┐
│  TemporalNarrativeGraph       │   Episodes become nodes, temporal proximity
│  (temporal_narrative_graph)   │   creates directed edges between them
└─────────┬────────────────────┘
          │
          ▼
┌──────────────────────────────┐
│  Graph Metrics                │   networkx computes centrality, transitions,
│  (graph_analysis.py)          │   cycles, density, and distribution
└─────────┬────────────────────┘
          │
          ▼
┌──────────────────┐       ┌──────────────────────┐
│  JSON API        │       │  Dashboard Page       │
│  /api/analytics/ │◄─────►│  D3.js + Chart.js     │
│  graph-metrics/  │       │  /dashboard/.../       │
└──────────────────┘       │  narrative             │
                           └──────────────────────┘
```

---

## Data Flow

### Step 1 — EmotionTimeline

**Module:** `dreamsApp/app/builder.py`

Takes raw MongoDB post documents and converts them into an immutable `EmotionTimeline` — a sorted tuple of `EmotionEvent` objects.

```python
EmotionEvent(
    timestamp=datetime(2026, 2, 16, 14, 18, 20),
    emotion_label="negative",      # from sentiment model
    score=0.95,                    # confidence
    source_id="67b1a2c3..."        # MongoDB _id
)
```

Each post becomes one `EmotionEvent`. The timeline is sorted by timestamp.

### Step 2 — Episode Segmentation

**Module:** `dreamsApp/analytics/episode_segmentation.py`

Splits the timeline into **episodes** using a time-gap threshold.

- **`gap_threshold = 24 hours`** (default)
- If two consecutive events are more than 24 hours apart → new episode starts
- If they're within 24 hours → same episode

Example: 5 posts over 3 days with a 2-day gap in the middle → 2 episodes.

Each `Episode` is an immutable dataclass:
```python
Episode(
    start_time=datetime(...),
    end_time=datetime(...),
    events=(EmotionEvent(...), EmotionEvent(...), ...),
)
```

### Step 3 — Narrative Graph Construction

**Module:** `dreamsApp/analytics/temporal_narrative_graph.py`

Builds a directed graph where:
- **Nodes** = episodes
- **Edges** = connections between episodes that are temporally close

Edge creation rule:
- For every pair of episodes (i, j) where i < j
- Classify their temporal relationship using `adjacency_threshold`
- If they overlap or are adjacent (gap ≤ threshold) → create an edge
- Assign a **weight** (0.0–1.0) reflecting connection strength

The `adjacency_threshold` controls how far apart two episodes can be and still get connected.

- **`adjacency_threshold = 7 days`** (default)
- Episodes within 7 days of each other → connected
- Episodes more than 7 days apart → disconnected

Edge types (from `ProximityRelation`):
| Relation | Meaning |
|----------|--------|
| `overlapping` | The episodes overlap in time |
| `adjacent` | The gap between them is ≤ adjacency_threshold |
| `disjoint` | The gap is > adjacency_threshold (edge NOT created by default) |

#### Edge Weights

Each edge carries a `weight` field (0.0–1.0) that quantifies connection strength based on temporal proximity:

| Relation | Weight Formula | Intuition |
|----------|---------------|----------|
| `overlapping` | Always `1.0` | Simultaneous episodes are maximally connected |
| `adjacent` | `1.0 − gap / threshold` | Linearly decreases as the gap grows; touching episodes get 1.0, episodes at the threshold get 0.0 |
| `disjoint` | Always `0.0` | No meaningful temporal connection |

The weight is computed by `_compute_edge_weight()` and validated to stay within `[0.0, 1.0]`.

The graph is always a **DAG** (directed acyclic graph) because edges only go from lower-index to higher-index episodes (earlier → later in time).

### Step 4 — Graph Analysis

**Module:** `dreamsApp/analytics/graph_analysis.py`

Converts the `TemporalNarrativeGraph` to a networkx `DiGraph` via `to_networkx()`, then computes three categories of metrics:

#### Graph Summary
| Metric | What it measures |
|--------|-----------------|
| `node_count` | Total number of episodes |
| `edge_count` | Total number of connections |
| `density` | How interconnected the graph is (0.0 = no edges, 1.0 = fully connected) |
| `connected_components` | Number of separate subgraphs (ideally 1 = continuous narrative) |
| `is_dag` | Always `true` by construction |

#### Node Metrics (per episode)
| Metric | What it measures |
|--------|-----------------|
| `degree_centrality` | Fraction of other nodes this episode is connected to |
| `in_degree_centrality` | How many earlier episodes connect to this one |
| `out_degree_centrality` | How many later episodes this one connects to |
| `betweenness_centrality` | How often this episode lies on shortest paths between others (weighted — shorter temporal gaps count as shorter paths) |
| `emotion_label` | Dominant emotion label across the episode's events |
| `event_count` | Number of posts in this episode |

#### Pattern Analysis
| Metric | What it measures |
|--------|-----------------|
| `common_transitions` | Top 10 most frequent emotion→emotion edge patterns (e.g., "negative → neutral: 3 times") |
| `emotional_cycles` | Recurring loops detected at the label level (e.g., [negative, positive] means the user oscillates between these) |
| `label_distribution` | Count of episodes per emotion (e.g., `{"positive": 7, "negative": 4, "neutral": 1}`) |

**Note on cycle detection:** The structural graph is a DAG, so it has no cycles by definition. Instead, cycles are detected on a **label-level graph** — a smaller graph where each unique emotion label is a node, and edges represent transitions that occurred. `nx.simple_cycles` finds loops in this label graph.

### Step 5 — Edges

The raw directed edge list is also returned so the frontend can render the exact graph structure:

```json
{
    "source": 3,
    "target": 4,
    "relation": "adjacent",
    "weight": 0.7143
}
```

The `weight` field (0.0–1.0) is rounded to 4 decimal places. It is used by both `betweenness_centrality` computation and the D3 visualization.

---

## Analytics Module

### File: `dreamsApp/analytics/graph_analysis.py`

**Public API:**
```python
from dreamsApp.analytics.graph_analysis import analyze_narrative_graph

result = analyze_narrative_graph(narrative_graph)
```

**Input:** A `TemporalNarrativeGraph` instance.

**Output:** A JSON-serializable dictionary:
```json
{
    "graph_summary": {
        "node_count": 12,
        "edge_count": 9,
        "density": 0.136,
        "connected_components": 2,
        "is_dag": true
    },
    "node_metrics": [
        {
            "node_index": 0,
            "emotion_label": "positive",
            "start_time": "2026-01-07T11:30:00",
            "end_time": "2026-01-07T11:48:00.000001",
            "event_count": 5,
            "degree_centrality": 0.182,
            "in_degree_centrality": 0.0,
            "out_degree_centrality": 0.182,
            "betweenness_centrality": 0.0
        }
    ],
    "edges": [
        {
            "source": 0,
            "target": 1,
            "relation": "adjacent",
            "weight": 0.8571
        }
    ],
    "pattern_analysis": {
        "common_transitions": [
            {
                "source_emotion": "positive",
                "target_emotion": "negative",
                "count": 3
            }
        ],
        "emotional_cycles": [
            ["negative", "positive"],
            ["negative", "neutral", "positive"]
        ],
        "label_distribution": {
            "positive": 7,
            "negative": 4,
            "neutral": 1
        }
    }
}
```

**Error handling:**
- `TypeError` if input is not a `TemporalNarrativeGraph`
- Empty graph returns a zeroed-out response (no crash)

**Design decisions:**
- Pure function — no Flask, no database, no side effects
- `networkx` imported lazily inside `to_networkx()` to avoid hard module-level dependency
- Limits: max 10 transitions, max 20 cycles

---

## API Endpoint

### `GET /api/analytics/graph-metrics/<user_id>`

**Authentication:** Login required

**Parameters:**
| Parameter | Type | Location | Description |
|-----------|------|----------|-------------|
| `user_id` | string | URL path | The user whose posts to analyze |

**Responses:**

| Status | Body | When |
|--------|------|------|
| `200` | `{"user_id": "...", "metrics": {...}}` | Analysis completed |
| `302` | Redirect to `/auth/login` | Request is not authenticated |
| `400` | `{"error": "Invalid user_id format"}` | `user_id` contains illegal characters or exceeds 64 chars |
| `404` | `{"error": "No posts found for user", "user_id": "..."}` | No posts exist for this user_id |
| `500` | `{"error": "Failed to compute graph metrics"}` | Server error (details logged, not exposed) |

**Internal pipeline:**
1. Fetch all posts for `user_id` from MongoDB, sorted by timestamp
2. Transform each post into a record: `{timestamp, emotion_label, score, source_id}`
3. `build_emotion_timeline()` → `segment_timeline_to_episodes()` → `build_narrative_graph()` → `analyze_narrative_graph()`
4. Return result as JSON

**File:** `dreamsApp/app/analytics/routes.py`

**Blueprint:** `analytics_api` registered at `/api/analytics`

---

## Dashboard Visualization

### URL: `/dashboard/user/<user_id>/narrative`

**Authentication:** Requires login (served via the `dashboard` blueprint which has `@login_required`).

**How to access:** From the user's profile page (`/dashboard/user/<user_id>`), click the "Narrative Structure Analysis" card at the top.

### Page Sections

#### 1. Summary Cards
Four cards showing at-a-glance metrics:
- **Episodes** — total node count
- **Connections** — total edge count
- **Density** — graph density (0–1)
- **Components** — number of disconnected subgraphs

#### 2. Interactive Force-Directed Graph (D3.js v7)
- Each **node** = one episode
- **Node color**: teal-green (positive), coral-red (negative), blue-grey (neutral)
- **Node size**: proportional to betweenness centrality (bigger = more important as a bridge)
- **Node effects**: drop shadow for depth; glow effect on hover; stroke color derived from emotion
- **Edges**: curved arc paths with directed arrows showing temporal flow
- **Edge thickness**: proportional to edge weight (stronger temporal connection = thicker line, 0.6–2.5px)
- **Edge opacity**: proportional to edge weight (0.25–0.80)
- **Edge color**: tinted by source node's emotion color
- **Arrow markers**: sleek small arrows; lighter variant for low-weight edges (≤ 0.4)
- **Draggable** nodes — user can rearrange the layout
- **Hover tooltips**: episode index, emotion, event count, timestamps, centrality values
- **Adaptive layout**: link distance and charge repulsion scale with node count for consistent spacing

#### 3. Centrality Bar Chart (Chart.js 4)
- Horizontal bar chart showing betweenness centrality per episode
- Bars colored by emotion
- Helps identify which episodes are narrative "hubs"

#### 4. Emotion Distribution Donut (Chart.js 4)
- Donut chart showing proportion of positive/negative/neutral episodes
- Quick visual balance indicator

#### 5. Common Transitions Table
- Ranked table of most frequent emotion→emotion transitions
- Columns: Rank, Transition (with colored arrows), Occurrences

#### 6. Emotional Cycles
- Displayed as pill badges showing cycle paths
- Example: `Negative → Positive → Negative` (loop closed visually)
- Indicates recurring emotional patterns

### Technical Details
- Page loads with a spinner, then fetches from `/api/analytics/graph-metrics/<user_id>` via `fetch()`
- All rendering is **client-side** (D3 + Chart.js) — no matplotlib blocking
- Uses CDN-loaded libraries (no local static files needed)
- Dark theme with radial gradient background for the graph container
- SVG filters (`feDropShadow`, `feGaussianBlur`) for node depth and hover glow
- D3 force simulation uses adaptive parameters: `velocityDecay(0.35)`, `alphaDecay(0.02)` for smooth settling

---

## Configuration

### Thresholds (in `routes.py`)

| Threshold | Default | Purpose |
|-----------|---------|---------|
| `DEFAULT_GAP_THRESHOLD` | `24 hours` | Posts further apart than this form separate episodes |
| `DEFAULT_ADJACENCY_THRESHOLD` | `7 days` | Episodes closer than this get connected by an edge |

**Why 7 days for adjacency?** Real-world users don't post every hour — they post daily or every few days. With 1 hour (the original default), almost every episode ended up isolated with zero edges. 7 days ensures a connected graph for typical posting patterns while still separating genuinely distinct narrative arcs (e.g., a 2-week gap = different storyline).

These are currently hardcoded but can be made configurable via `app.config` in the future.

### Dependency

`networkx>=3.0` — added to `requirements.txt`. Used for graph construction (DiGraph), centrality computation, and cycle detection.

---

## File Map

### Analytics Layer (no Flask dependency)
| File | Role |
|------|------|
| `dreamsApp/analytics/emotion_timeline.py` | `EmotionEvent`, `EmotionTimeline` — immutable data containers |
| `dreamsApp/analytics/emotion_episode.py` | `Episode` — immutable episode container |
| `dreamsApp/analytics/emotion_segmentation.py` | Time-gap segmentation logic |
| `dreamsApp/analytics/episode_segmentation.py` | `segment_timeline_to_episodes()` |
| `dreamsApp/analytics/episode_proximity.py` | `ProximityRelation`, gap/overlap computation |
| `dreamsApp/analytics/temporal_narrative_graph.py` | `TemporalNarrativeGraph`, `build_narrative_graph()`, `to_networkx()` |
| `dreamsApp/analytics/graph_analysis.py` | `analyze_narrative_graph()` — all graph metric computation |

### App Layer (Flask)
| File | Role |
|------|------|
| `dreamsApp/app/builder.py` | `build_emotion_timeline()` — converts raw records to timeline |
| `dreamsApp/app/analytics/__init__.py` | Blueprint: `analytics_api` at `/api/analytics` |
| `dreamsApp/app/analytics/routes.py` | `GET /api/analytics/graph-metrics/<user_id>` handler |
| `dreamsApp/app/__init__.py` | Registers the analytics blueprint in `create_app()` |
| `dreamsApp/app/dashboard/main.py` | `narrative()` route for the visualization page |

### Templates
| File | Role |
|------|------|
| `dreamsApp/app/templates/dashboard/narrative.html` | Full visualization page (D3 + Chart.js) |
| `dreamsApp/app/templates/dashboard/profile.html` | Modified — added link card to narrative page |

### Tests
| File | Test Count | Scope |
|------|-----------|-------|
| `tests/test_graph_analysis.py` | 33 | Unit tests for all metric computations + edges + weighted edges + to_networkx() |
| `tests/test_temporal_narrative_graph.py` | 41 | Unit tests for episodes, proximity, narrative edges, edge weights, graph construction |
| `tests/test_graph_metrics_api.py` | 9 | Integration tests for API endpoint + dashboard route |

---

## Testing

### Run all related tests
```bash
python -m pytest tests/test_graph_analysis.py tests/test_graph_metrics_api.py -v
```

### Run the full regression suite
```bash
python -m pytest tests/test_graph_analysis.py tests/test_graph_metrics_api.py tests/test_temporal_narrative_graph.py tests/test_timeline.py -v
```

### Test coverage breakdown

**Unit tests (`test_graph_analysis.py` — 33 tests):**
- Input validation (TypeError on wrong types, empty graph returns zeroed response)
- Graph summary accuracy (density, components, edge count)
- Node metrics (centrality values, emotion label derivation, ISO timestamp format)
- Pattern analysis (transition sorting, cycle detection, label distribution)
- Edge response (count matches summary, correct structure with weight field, DAG ordering)
- `to_networkx()` (node/edge counts, attributes, edge weight propagation, empty graph handling)

**Unit tests (`test_temporal_narrative_graph.py` — 41 tests):**
- Episode immutability and temporal invariants
- Episode segmentation (empty, single, gap-based splitting)
- Episode proximity (overlap, gap, adjacency classification)
- NarrativeEdge validation (ordering, index bounds, weight range 0.0–1.0)
- Edge weight computation (overlapping→1.0, touching→1.0, gap decay, disjoint→0.0)
- Graph construction (deterministic, immutable, node/edge queries)
- Weight serialization in `to_dict()`

**Integration tests (`test_graph_metrics_api.py` — 9 tests):**
- 404 for missing user
- 200 with valid data + correct JSON structure
- All response sections present (graph_summary, node_metrics, edges, pattern_analysis)
- Unauthenticated requests redirect to login (302)
- Invalid `user_id` (bad chars or length > 64) returns 400
- Narrative page returns 200
- Narrative page contains D3 and Chart.js references
- Narrative page contains all expected DOM elements
- Narrative page includes the user_id

### Manual testing with Postman

1. Upload 8–10 posts to `POST /api/upload` with varied captions and timestamps
2. Hit `GET /api/analytics/graph-metrics/<user_id>`
3. Verify JSON structure matches the schema above
4. Open `/dashboard/user/<user_id>/narrative` in a browser (logged in)
5. Verify all 6 visual sections render correctly

---

## Known Limitations

1. **Only 3 emotion labels** — The sentiment model (RoBERTa) only produces `positive`, `negative`, `neutral`. Transitions and cycles are limited to these 3 labels. When finer-grained models (e.g., CHIME dimensions) are added, the analysis benefits automatically.

2. **No structural cycles** — The graph is always a DAG because edges go from earlier to later episodes only. Cycles are detected at the label level instead.

3. **Thresholds are not configurable per-request** — `gap_threshold` and `adjacency_threshold` are currently hardcoded. A future iteration could accept them as query parameters.

4. **D3 graph edge approximation** — The D3 visualization uses the exact edge list from the API, but the force-directed layout may not perfectly represent temporal ordering visually (nodes may float to positions that don't reflect chronological order).

5. **No caching** — Every request recomputes the full pipeline from MongoDB. For users with many posts, this could be slow. A caching layer could be added.

6. **Computation is O(N²)** — The `build_narrative_graph` function checks all pairs of episodes. For users with hundreds of episodes, this becomes slow. Currently fine for typical usage (10–50 posts).

7. **Weights are purely temporal** — Edge weights are based solely on time-gap proximity. Semantic similarity between episode content is not yet factored in. A future "temporo-semantic" model could combine temporal weight with text embedding cosine similarity for richer connections.

---

## Future Roadmap

| Phase | Feature | Value |
|-------|---------|-------|
| **Next** | Query parameter overrides for thresholds | Researchers can tune gap/adjacency per user |
| **Next** | Caching layer for computed metrics | Faster repeated access |
| **Phase 2** | Cohort comparison view | Compare graph metrics across user groups |
| **Phase 2** | Longitudinal tracking | Track how a user's graph structure changes over months |
| **Phase 3** | Alert system for negative cycles | Flag users stuck in recurring negative loops |
| **Phase 3** | Graph metrics as ML features | Feed centrality/density into predictive models |
| **Phase 3** | CHIME-dimension integration | Use CHIME labels instead of just positive/negative/neutral for richer analysis |
| **Phase 3** | Semantic similarity edges | Combine temporal proximity with text embedding cosine similarity for temporo-semantic edge weights |
| **Phase 4** | Research data export (CSV/GraphML) | Researchers can download graph data for external analysis and papers |
