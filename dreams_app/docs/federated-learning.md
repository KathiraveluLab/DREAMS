# Federated Learning - Self-Correcting CHIME Model

## Overview

The DREAMS application includes a **Federated Learning (FL)** feature that enables the CHIME classification model to improve over time based on user feedback. This creates a self-correcting AI system that adapts to real-world usage patterns while maintaining privacy and safety.

### Key Benefits

- **Autonomous Improvement**: The model gets smarter without manual intervention.
- **Privacy-Preserving**: User data is used for training locally; only model updates are persisted.
- **Safety-First**: Validation gates prevent model degradation.
- **Zero Maintenance**: Event-driven architecture requires no scheduled jobs or external services.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              USER LAYER                                  │
├─────────────────────────────────────────────────────────────────────────┤
│   User uploads post → Gets AI prediction → Corrects if wrong            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           WEB SERVER (Flask)                             │
├─────────────────────────────────────────────────────────────────────────┤
│   /correct_chime endpoint:                                               │
│   1. Saves correction to MongoDB (adds to the FL queue).                 │
│   2. Counts unprocessed records; if >= 50 and the FL lock is free,       │
│      claims it and starts a single training thread. Otherwise the        │
│      correction waits in the queue for the next round.                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌──────────────────────────────┐    ┌──────────────────────────────────────┐
│         MongoDB              │    │       FL Worker (Background)          │
├──────────────────────────────┤    ├──────────────────────────────────────┤
│ posts collection:            │    │ 1. Fetch 50 corrections               │
│ - corrected_label            │    │ 2. Load previous model state          │
│ - is_fl_processed            │    │ 3. Train classifier head (freeze base)│
│ - correction_timestamp       │    │ 4. Validate (Anchor + Improvement)    │
│ - fl_round_date              │    │ 5. If pass: Update production model   │
└──────────────────────────────┘    │ 6. Mark corrections as processed      │
                                    └──────────────────────────────────────┘
                                                    │
                                                    ▼
                                    ┌──────────────────────────────────────┐
                                    │    Local File System                  │
                                    ├──────────────────────────────────────┤
                                    │ dreams_app/app/models/                 │
                                    │   └── production_chime_model/         │
                                    │       ├── config.json                 │
                                    │       ├── pytorch_model.bin           │
                                    │       └── tokenizer files             │
                                    └──────────────────────────────────────┘
                                                    │
                                                    ▼
                                    ┌──────────────────────────────────────┐
                                    │    Inference (sentiment.py)           │
                                    ├──────────────────────────────────────┤
                                    │ Loads production_chime_model if exists│
                                    │ Falls back to HuggingFace if missing  │
                                    └──────────────────────────────────────┘
```

---

## How It Works

### Step 1: User Feedback Collection

When a user views their post analysis on the dashboard, they can:
- **Accept** the AI's prediction (confirms it's correct).
- **Edit** the prediction by selecting the correct CHIME dimension.
- Select **"None"** if the text doesn't fit any CHIME category.

### Step 2: Database Storage

Each correction is stored in the `posts` collection with:

```javascript
{
  "_id": ObjectId("..."),
  "caption": "User's journal text",
  "chime_analysis": { "label": "Hope", "score": 0.85 },  // Original prediction
  "corrected_label": "Meaning",                          // User's correction
  "is_fl_processed": false,                              // Training status
  "correction_timestamp": ISODate("2026-01-30T...")
}
```

### Step 3: Training Trigger

After each correction is saved, the route counts how many documents still have `is_fl_processed: False`. When the queue reaches 50 and the `fl_training_lock` document is free, the handler atomically flips the lock, launches the worker in a daemon thread, and immediately returns to the browser. If another training round is already running, the lock acquisition fails and the correction simply remains in the queue until the next round.

### Step 4: Model Training

The FL Worker performs:

1. **Data Preparation**: Fetches 50 corrections, maps labels to IDs.
2. **Model Loading**: 
   - If `production_chime_model` exists → Load it (continuous learning).
   - If not → Load base model from HuggingFace.
3. **Training**:
   - Freezes BERT base layers (preserves pre-trained knowledge).
   - Trains only the classifier head (lightweight, fast).
   - Runs for 3 epochs with conservative learning rate (1e-5).
4. **Validation Gate**:
   - **Anchor Check**: Model must correctly classify 3/5 basic examples.
   - **Improvement Check**: Model must correctly classify 50% of training data.
5. **Deployment**:
   - If validation passes → Atomic swap to production folder.
   - If validation fails → Discard changes, log error.

### Step 5: Inference

The `sentiment.py` module automatically:
- Checks for `production_chime_model` on startup.
- Loads it if available (self-corrected model).
- Falls back to HuggingFace if not (base model).

---

## File Structure

```
dreams_app/
├── app/
│   ├── dashboard/
│   │   └── main.py              # /correct_chime endpoint with FL trigger
│   ├── models/
│   │   └── production_chime_model/  # Updated model (created after first training)
│   ├── core/
│   │   ├── logger.py            # Production logging setup
│   │   └── sentiment.py         # Model loading with local-first logic
│   └── fl_worker.py             # Core training logic
├── docs/
│   └── federated-learning.md    # This file
└── tests/
    └── test_fl.py               # End-to-end test script
```

---

## Configuration

All configuration is in `fl_worker.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 50 | Number of corrections before training triggers |
| `LEARNING_RATE` | 1e-5 | Conservative rate to preserve base knowledge |
| `EPOCHS` | 3 | Training iterations per round |
| `BASE_MODEL_ID` | `ashh007/dreams-chime-bert` | HuggingFace model for initial load |

### Anchor Examples

The validation gate uses 5 hardcoded examples to detect catastrophic forgetting:

```python
ANCHOR_EXAMPLES = [
    {"text": "I feel completely safe and surrounded.", "label": "Connectedness"},
    {"text": "I see a bright future ahead.", "label": "Hope"},
    {"text": "I don't know who I am anymore.", "label": "Identity"},
    {"text": "My life has deep purpose.", "label": "Meaning"},
    {"text": "I have the power to change my situation.", "label": "Empowerment"}
]
```

---

## Logging

All FL activity is logged to `logs/fl_worker.log`:

```
2026-01-30 14:32:15 | INFO     | FL WORKER: Waking up...
2026-01-30 14:32:16 | INFO     | Starting Training Round with 50 samples.
2026-01-30 14:32:45 | INFO     | [Epoch 1/3] Loss: 0.3421
2026-01-30 14:33:12 | INFO     | [Epoch 2/3] Loss: 0.2187
2026-01-30 14:33:38 | INFO     | [Epoch 3/3] Loss: 0.1543
2026-01-30 14:33:40 | INFO     | Running Validation Gate...
2026-01-30 14:33:42 | INFO     | [Safety Check] Anchor Accuracy: 5/5
2026-01-30 14:33:44 | INFO     | [Improvement Check] Training Set Accuracy: 47/50
2026-01-30 14:33:45 | INFO     | Update Accepted! Promoting to Production...
2026-01-30 14:33:46 | INFO     | SUCCESS: Central Model updated.
```

---

## Testing

### Automated Test

Run the end-to-end test:

```powershell
python tests/test_fl.py
```

This script:
1. Injects mock corrections into MongoDB.
2. Runs the FL worker.
3. Verifies database updates.
4. Verifies model folder creation.
5. Verifies inference uses the new model.
6. Cleans up test data.

### Manual Test

1. Start the Flask app.
2. Upload posts and correct predictions until you reach 50.
3. Check `logs/fl_worker.log` for training activity.
4. Verify `dreams_app/app/models/production_chime_model/` exists.

---

## Safety Mechanisms

### 1. Validation Gate

Prevents deploying a degraded model:
- **Anchor Check**: Must recognize basic examples (prevents catastrophic forgetting).
- **Improvement Check**: Must learn from new data (prevents useless updates).

### 2. Atomic Swap

Model files are written to a temp folder first, then moved atomically. If the process crashes mid-write, the production model remains intact.

### 3. Frozen Base Layers

Only the classifier head is trained. The BERT base (pre-trained on 2000+ examples) is never modified, preserving its knowledge.

### 4. Conservative Learning Rate

A low learning rate (1e-5) ensures the model nudges toward new data without forgetting old patterns.

---

## Future Enhancements

| Enhancement | Priority | Description |
|-------------|----------|-------------|
| Expand Anchor Set | Medium | Add 50+ examples for stronger safety checks |
| Model Versioning | Low | Backup old models before overwriting |
| Admin Dashboard | Low | UI to view training history and logs |
| Lock Monitoring | Low | Track `fl_training_lock` contention/latency to keep single-thread guarantee at scale |
| Outlier Detection | Low | Reject statistically anomalous corrections |

---

## Troubleshooting

### Training Never Triggers

- Check correction count: `db.posts.countDocuments({corrected_label: {$exists: true}, is_fl_processed: false})`
- Ensure threshold is 50 (check `BATCH_SIZE` in `fl_worker.py`).

### Validation Always Fails

- Check `logs/fl_worker.log` for anchor failures.
- Verify `label2id` mapping matches your model's config.
- Ensure training data quality (not all "None" labels).

### Model Not Loading

- Verify `production_chime_model/` contains `pytorch_model.bin` and `config.json`.
- Check `sentiment.py` logs for loading errors.
- Delete the folder to force fallback to HuggingFace.

### Memory Issues

- Training runs on CPU by default.
- If OOM occurs, reduce `BATCH_SIZE` or run on a machine with more RAM.

---

## API Reference

### POST /dashboard/correct_chime

Submit a correction for a post's CHIME classification.

**Request Body:**
```json
{
  "post_id": "ObjectId string",
  "corrected_label": "Hope|Connectedness|Identity|Meaning|Empowerment|None"
}
```

**Response:**
```json
{
  "success": true
}
```

**Side Effects:**
- Stores correction in database.
- Triggers FL training if 50+ corrections pending.

**Notes:**
- The backend relies on `current_user`/session identity, so no `user_id` value is required in the payload.

---

## References

- [CHIME Recovery Framework](https://en.wikipedia.org/wiki/Recovery_model)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Federated Learning Overview](https://federated.withgoogle.com/)
- [Base Model: ashh007/dreams-chime-bert](https://huggingface.co/ashh007/dreams-chime-bert)
