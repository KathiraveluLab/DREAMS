from transformers import pipeline
import json
import os

# Load emotion analysis model
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

# Custom Paths 
input_dir = "/media/bagwan/BAGWAN/workspace/DREAMS/dream-integration/data/person-01/sample-01"         
output_dir = "/media/bagwan/BAGWAN/workspace/DREAMS/dream-integration/data/person-01/analysis-p01"       
os.makedirs(output_dir, exist_ok=True)

# Files to analyse
files_to_analyze = ["clip-01.txt", "description-01.txt"]

output = {}

for file_name in files_to_analyze:
    file_path = os.path.join(input_dir, file_name)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        scores = classifier(text)
        output[file_name] = scores[0]  # each score is a list of dicts

# Save result
output_file_path = os.path.join(output_dir, "text_scores.json")
with open(output_file_path, 'w', encoding='utf-8') as f: 
    json.dump(output, f, indent=2)

print(f"Text analysis done. Results saved to {output_file_path}")