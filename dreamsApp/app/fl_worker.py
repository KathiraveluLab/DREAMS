import torch
import shutil
import os
import datetime
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from dreamsApp.app import create_app
from dreamsApp.app.utils.logger import setup_logger

# Setup Logger
logger = setup_logger('fl_worker')

# --- CONFIGURATION ---
BASE_MODEL_ID = "ashh007/dreams-chime-bert"
# Determine absolute paths based on app location to ensure robustness
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# dreamsApp/app/models/production_chime_model
PRODUCTION_MODEL_DIR = os.path.join(BASE_DIR, "models", "production_chime_model")
# dreamsApp/app/models/temp_training_artifact
TEMP_MODEL_DIR = os.path.join(BASE_DIR, "models", "temp_training_artifact")

BATCH_SIZE = 50
LEARNING_RATE = 1e-5 # Conservative learning rate

# "Anchor Set": 5 obvious examples that MUST remain correct (Prevent catastrophic forgetting)
ANCHOR_EXAMPLES = [
    {"text": "I feel completely safe and surrounded.", "label": "Connectedness"}, 
    {"text": "I see a bright future ahead.", "label": "Hope"},         
    {"text": "I don't know who I am anymore.", "label": "Identity"},       
    {"text": "My life has deep purpose.", "label": "Meaning"},             
    {"text": "I have the power to change my situation.", "label": "Empowerment"} 
]

def validate_model(model, tokenizer, training_samples, label2id):
    """
    Returns True if model passes BOTH Safety Checks and Improvement Checks.
    """
    model.eval()
    logger.info("Running Validation Gate...")

    # 1. ANCHOR CHECK (Safety)
    correct_anchors = 0
    with torch.no_grad():
        for example in ANCHOR_EXAMPLES:
            inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            pred_id = torch.argmax(outputs.logits).item()
            
            # Dynamic Label Check
            target_str = example["label"]
            target_id = label2id.get(target_str)
            
            # If the label exists in model config and matches prediction
            if target_id is not None and pred_id == target_id:
                correct_anchors += 1
            else:
                # Debug print for failure
                # Get the string label for the prediction
                id2label = {v: k for k, v in label2id.items()}
                pred_str = id2label.get(pred_id, "Unknown")
                # SECURITY NOTE: Only logging hardcoded anchor examples, not user data
                logger.debug(f"[Anchor Fail] Text: '{example['text'][:30]}...' Expected: {target_str}, Got: {pred_str}")

    logger.info(f"[Safety Check] Anchor Accuracy: {correct_anchors}/{len(ANCHOR_EXAMPLES)}")
    if correct_anchors < 4: # Stricter check for catastrophic forgetting
        logger.error("FAIL: Model has forgotten basic concepts (Catastrophic Forgetting).")
        return False

    # 2. IMPROVEMENT CHECK (Did it learn?)
    correct_new = 0
    total_new = len(training_samples)
    with torch.no_grad():
        for text, label_idx in training_samples:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            pred_label = torch.argmax(outputs.logits).item()
            if pred_label == label_idx:
                correct_new += 1
    
    logger.info(f"[Improvement Check] Training Set Accuracy: {correct_new}/{total_new}")
    
    if total_new > 0 and correct_new / total_new < 0.5:
        logger.error("FAIL: Model failed to learn the new corrections.")
        return False

    return True

def run_federated_round():
    app = create_app()
    with app.app_context():
        mongo = app.mongo
        logger.info("FL WORKER: Waking up...")
        
        try:
            # CLEANUP: Reset any stale 'processing' documents (older than 1 hour)
            one_hour_ago = datetime.datetime.now() - datetime.timedelta(hours=1)
            stale_reset = mongo['posts'].update_many(
                {
                    'is_fl_processed': 'processing',
                    'processing_started_at': {'$lt': one_hour_ago}
                },
                {'$set': {'is_fl_processed': False}, '$unset': {'processing_started_at': ''}}
            )
            if stale_reset.modified_count > 0:
                logger.warning(f"Reset {stale_reset.modified_count} stale 'processing' documents.")
            
            # 1. Atomically CLAIM Pending Data (Prevents Race Condition)
            # Step 1a: Find IDs of documents to claim
            query = {
                'corrected_label': {'$exists': True},
                'is_fl_processed': False  # Only unclaimed documents
            }
            
            candidate_ids = [doc['_id'] for doc in mongo['posts'].find(query, {'_id': 1}).limit(BATCH_SIZE)]
            
            if len(candidate_ids) < BATCH_SIZE:
                logger.info(f"Only {len(candidate_ids)} corrections available. Waiting for {BATCH_SIZE}.")
                return
            
            # Step 1b: Atomically claim these documents by setting status to 'processing'
            claim_result = mongo['posts'].update_many(
                {'_id': {'$in': candidate_ids}, 'is_fl_processed': False},  # Re-check status!
                {'$set': {
                    'is_fl_processed': 'processing',
                    'processing_started_at': datetime.datetime.now()
                }}
            )
            
            if claim_result.modified_count < BATCH_SIZE:
                logger.warning(f"Race condition detected: Only claimed {claim_result.modified_count}/{BATCH_SIZE} documents. Another worker may be running. Aborting.")
                # Release any documents we did claim back to 'False'
                mongo['posts'].update_many(
                    {'_id': {'$in': candidate_ids}, 'is_fl_processed': 'processing'},
                    {'$set': {'is_fl_processed': False}, '$unset': {'processing_started_at': ''}}
                )
                return
            
            # Step 1c: Now fetch the full documents we successfully claimed
            pending_posts = list(mongo['posts'].find({'_id': {'$in': candidate_ids}, 'is_fl_processed': 'processing'}))
            logger.info(f"Successfully claimed {len(pending_posts)} documents for training.")

            # Prepare Data
            # We need to fetch the configuration to know the label map
            try:
                config = AutoConfig.from_pretrained(BASE_MODEL_ID)
                label2id = config.label2id
            except Exception as e:
                # Fallback if config fetch fails
                logger.warning(f"Could not load config from HuggingFace: {e}. Using fallback label map.")
                label2id = {"Connectedness": 0, "Hope": 1, "Identity": 2, "Meaning": 3, "Empowerment": 4}

            training_data = [] # List of (text, label_idx)
            valid_ids = []

            # SECURITY: Do not log the caption/text content to avoid exposing user data
            for p in pending_posts:
                lbl = p.get('corrected_label')
                if lbl in label2id:
                    training_data.append((p.get('caption'), label2id[lbl]))
                    valid_ids.append(p['_id'])
                elif lbl == 'None':
                    # Mark 'None' as processed but don't train
                    mongo['posts'].update_one({'_id': p['_id']}, {'$set': {'is_fl_processed': True, 'fl_status': 'skipped'}})
                    # Log only the document ID, not the content
                    logger.debug(f"Skipped 'None' label for post {p['_id']}")

            if not training_data:
                logger.info("No valid labels found (mostly 'None'). Marking processed and exiting.")
                return

            # SECURITY: Only log counts/statistics, never actual user text
            logger.info(f"Starting Training Round with {len(training_data)} samples.")

            # 2. Load Model (CONTINUOUS LEARNING)
            if os.path.exists(PRODUCTION_MODEL_DIR):
                logger.info(f"Loading existing Production Model from {PRODUCTION_MODEL_DIR}...")
                load_path = PRODUCTION_MODEL_DIR
            else:
                logger.info("First run: Loading Base Model from Hugging Face...")
                load_path = BASE_MODEL_ID

            tokenizer = AutoTokenizer.from_pretrained(load_path)
            model = AutoModelForSequenceClassification.from_pretrained(load_path, num_labels=len(label2id))

            # Freeze BERT Base, Train Head
            if hasattr(model, 'bert'):
                for param in model.bert.parameters():
                    param.requires_grad = False
            elif hasattr(model, 'base_model'):
                for param in model.base_model.parameters():
                    param.requires_grad = False
            
            logger.debug("Base layers frozen. Training classifier head only.")
            
            # 3. Training Loop
            model.train()
            optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE)
            
            texts = [item[0] for item in training_data]
            labels_tensor = torch.tensor([item[1] for item in training_data])
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

            EPOCHS = 3
            for epoch in range(EPOCHS):
                optimizer.zero_grad()
                outputs = model(**inputs, labels=labels_tensor)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                logger.info(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {loss.item():.4f}")

            # 4. Save to TEMP
            if os.path.exists(TEMP_MODEL_DIR):
                shutil.rmtree(TEMP_MODEL_DIR) # Clean start
            model.save_pretrained(TEMP_MODEL_DIR)
            tokenizer.save_pretrained(TEMP_MODEL_DIR)
            logger.debug(f"Model saved to temp directory: {TEMP_MODEL_DIR}")

            # 5. Validation Gate
            passed = validate_model(model, tokenizer, training_data, label2id)

            if passed:
                logger.info("Update Accepted! Promoting to Production...")
                # ATOMIC SWAP: Overwrite production folder
                if os.path.exists(PRODUCTION_MODEL_DIR):
                    shutil.rmtree(PRODUCTION_MODEL_DIR)
                
                # Ensure parent dict exists
                os.makedirs(os.path.dirname(PRODUCTION_MODEL_DIR), exist_ok=True)
                
                shutil.copytree(TEMP_MODEL_DIR, PRODUCTION_MODEL_DIR)
                logger.info(f"SUCCESS: Central Model updated at {PRODUCTION_MODEL_DIR}")
            else:
                logger.warning("Update Rejected. Discarding changes.")
            
            # Cleanup Temp
            if os.path.exists(TEMP_MODEL_DIR):
                shutil.rmtree(TEMP_MODEL_DIR)

            # 6. Finish
            logger.info("Updating database records...")
            mongo['posts'].update_many(
                {'_id': {'$in': valid_ids}},
                {
                    '$set': {
                        'is_fl_processed': True,  # Mark as fully processed (was 'processing')
                        'fl_round_date': datetime.datetime.now()
                    },
                    '$unset': {'processing_started_at': ''}  # Clean up temp field
                }
            )
            logger.info(f"Round Successfully Completed. Processed {len(valid_ids)} items.")
            
        except Exception as e:
            logger.error(f"CRITICAL FAILURE during FL round: {str(e)}", exc_info=True)
            # Cleanup temp if it exists after a failure
            if os.path.exists(TEMP_MODEL_DIR):
                shutil.rmtree(TEMP_MODEL_DIR)
            # Release any documents we claimed back to the queue
            try:
                mongo['posts'].update_many(
                    {'is_fl_processed': 'processing'},
                    {'$set': {'is_fl_processed': False}, '$unset': {'processing_started_at': ''}}
                )
                logger.info("Released claimed documents back to queue after failure.")
            except:
                pass  # Best effort release


# Allow running as standalone script for manual testing
if __name__ == "__main__":
    run_federated_round()
