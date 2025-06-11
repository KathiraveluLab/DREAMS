import functools, json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from flask import Blueprint, jsonify, flash, redirect, request, session, url_for

'''The model will be replaced with our custom model after we train it currently it uses distillBert we can
fine tune this model and replace the last layer to get custom classification  Fine tunning this model will improve this model
and relevance'''

model_path = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

bp = Blueprint('api', __name__, url_prefix='/sentiments')

@bp.route('/caption', methods=['POST'])
def sentimentAnalysis():
    data = request.get_json()
    caption = data.get('caption')


    if not isinstance(caption, str):
        return jsonify({'error': 'caption is not string'}), 400
    
    if len(caption) > 500:
        return jsonify({'error': 'caption is too long'}), 400

    ## loading the model from hugging face
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    if not caption:
        return jsonify({'error': 'No caption provided'}), 400
    
    try:
        inputs = tokenizer(caption, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Processing the logits to get the output
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        # Get the predicted class and score
        predicted_class = torch.argmax(scores, dim=1).item()
        predicted_score = scores[0][predicted_class].item()
        
        # Map the class ID to label (adjust these mappings to match your model's output classes)
        id2label = {
            0: "NEGATIVE",
            1: "POSITIVE",
        }
        
        sentiment_label = id2label.get(predicted_class, "UNKNOWN")
        
        return jsonify([
            {
                "label": sentiment_label,
                "score": predicted_score
            }
        ])
    except Exception as e:
        return jsonify({'error':str(e)}), 500
    
    