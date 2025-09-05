from transformers import pipeline
import json
import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run text emotion analysis")
parser.add_argument("--transcript", type=str, help="Path to transcript file", required=False)
parser.add_argument("--description", type=str, help="Path to description file", required=False)
parser.add_argument("--output", type=str, help="Directory to save output JSON", required=True)
args = parser.parse_args()

# Load emotion analysis model
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

# Files to analyse (only those that exist)
files_to_analyze = []
if args.transcript and os.path.exists(args.transcript):
    files_to_analyze.append(args.transcript)
if args.description and os.path.exists(args.description):
    files_to_analyze.append(args.description)

output = {}

for file_path in files_to_analyze:
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    scores = classifier(text)
    output[os.path.basename(file_path)] = scores[0]  # keep filename as key

# Ensure output dir exists
os.makedirs(args.output, exist_ok=True)

# Save result
output_file_path = os.path.join(args.output, "text_scores.json")
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2)

print(f"Text analysis done. Results saved to {output_file_path}")