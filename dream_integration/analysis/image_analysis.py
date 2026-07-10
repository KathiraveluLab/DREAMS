import argparse
import os
import json
from deepface import DeepFace 

def analyze_image_emotion(image_path, output_path):
    # Analyze only emotion
    result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)

    if isinstance(result, list):
        result = result[0]

    # Converting emotion scores to standard float (for json Serialisation)
    emotion_scores = {emotion: float(score) for emotion, score in result["emotion"].items()}

    output = {
        "dominant_emotion": result["dominant_emotion"],
        "emotion_scores": emotion_scores
    }

    # Save output as JSON
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f" Image emotion analysis complete. Output saved at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze emotion in an image using DeepFace")
    parser.add_argument('--image', required=True, help='Input Path')
    parser.add_argument('--output', required=True, help='Output path')

    args = parser.parse_args()
    analyze_image_emotion(args.image, args.output)