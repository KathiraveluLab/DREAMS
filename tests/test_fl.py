import sys
import os
import datetime
from bson.objectid import ObjectId

# Add the project root to the python path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dreamsApp.app import create_app
from dreamsApp.app.fl_worker import run_federated_round

def test_fl_loop():
    app = create_app()
    with app.app_context():
        mongo = app.mongo
        collection = mongo['posts']
        
        print(">>> TEST: setting up mock data...")
        
        # 1. Create Mock Data
        # We need at least 10 entries to trigger the worker logic (assuming BATCH_SIZE=10 in fl_worker.py)
        # We'll create 5 "Hope" and 5 "Meaning" corrections, plus 1 "None" to test skipping.
        
        test_ids = []
        
        mock_posts = []
        
        # Batch 1: Valid Corrections
        for i in range(5):
            mock_posts.append({
                'user_id': 'test_user_automated',
                'caption': f'This is a test caption related to hope {i}',
                'timestamp': datetime.datetime.now(),
                'chime_analysis': {'label': 'Connectedness'}, # Originally wrong
                'corrected_label': 'Hope', # User corrected it
                'is_fl_processed': False
            })
            
        for i in range(4): # Reduced to 4 to make total batch size matches worker limit (10)
            mock_posts.append({
                'user_id': 'test_user_automated',
                'caption': f'This is a test caption related to meaning {i}',
                'timestamp': datetime.datetime.now(),
                'chime_analysis': {'label': 'Connectedness'}, 
                'corrected_label': 'Meaning',
                'is_fl_processed': False
            })
            
        # Batch 2: Skipped Correction
        mock_posts.append({
            'user_id': 'test_user_automated',
            'caption': 'This is the worst day ever',
            'timestamp': datetime.datetime.now(),
            'chime_analysis': {'label': 'Connectedness'}, 
            'corrected_label': 'None',
            'is_fl_processed': False
        })

        # Insert
        result = collection.insert_many(mock_posts)
        test_ids = result.inserted_ids
        print(f">>> TEST: Inserted {len(test_ids)} mock documents.")

        # 2. Run the Worker
        print("\n>>> TEST: Running FL Worker Step...")
        try:
            run_federated_round()
        except Exception as e:
            print(f"!!! TEST FAILED: Worker crashed with error: {e}")
            # Cleanup
            collection.delete_many({'_id': {'$in': test_ids}})
            return

        # 3. Verify Results
        print("\n>>> TEST: Verifying DB Updates...")
        
        # Check valid posts
        processed_count = collection.count_documents({
            '_id': {'$in': test_ids},
            'is_fl_processed': True
        })
        
        print(f"    processed_count: {processed_count} (Expected: {len(test_ids)})")
        
        if processed_count == len(test_ids):
            print(">>> TEST SUCCESS: All documents were processed.")
        else:
            print("!!! TEST FAILED: Some documents were not processed.")
            
        # Check if the skipped one has the specific status
        skipped_doc = collection.find_one({'corrected_label': 'None', '_id': {'$in': test_ids}})
        if skipped_doc and skipped_doc.get('fl_status') == 'skipped':
             print(">>> TEST SUCCESS: 'None' label was correctly marked as skipped.")
        
        # 4. Verify Model Creation & Loading logic
        print("\n>>> TEST: Verifying Inference (End-to-End)...")
        from dreamsApp.app.utils.sentiment import SentimentAnalyzer
        
        # Check directory existence
        # Current file is in /tests, so we go up one level to root
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        prod_model_path = os.path.join(base_dir, "dreamsApp", "app", "models", "production_chime_model")
        
        if os.path.exists(prod_model_path):
            print(f">>> TEST SUCCESS: Production model folder created at {prod_model_path}")
            
            # Now verify the app loads it
            analyzer = SentimentAnalyzer()
            # Force reload to ensure we pick up the new file
            analyzer._chime_classifier = None 
            
            print("    Loading classifier (should pick up local model)...")
            result = analyzer.analyze_chime("I feel so hopeful about my future.")
            print(f"    Inference Result: {result}")
            
            if result and 'label' in result:
                 print(">>> TEST SUCCESS: Inference pipeline is working with the new model.")
            else:
                 print("!!! TEST FAILED: Inference pipeline returned invalid result.")
        else:
            print(f"!!! TEST FAILED: Production model folder NOT found at {prod_model_path}")

        # 5. Cleanup
        print("\n>>> TEST: Cleaning up mock data...")
        collection.delete_many({'_id': {'$in': test_ids}})
        print(">>> TEST: Done.")

if __name__ == "__main__":
    test_fl_loop()
