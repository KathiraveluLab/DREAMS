import os
import torch
import logging
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import requests
from flask import Blueprint, request, jsonify

# Globals for lazy loading
_blip_processor = None
_blip_model = None
_sentiment_tokenizer = None
_sentiment_config = None
_sentiment_model = None
_absa_model = None

def get_blip_models():
    global _blip_processor, _blip_model
    if _blip_processor is None or _blip_model is None:
        print("Loading Blip models...")
        _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        _blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return _blip_processor, _blip_model

def get_sentiment_models():
    global _sentiment_tokenizer, _sentiment_config, _sentiment_model
    if _sentiment_model is None:
        print("Loading Sentiment models...")
        sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        _sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
        _sentiment_config = AutoConfig.from_pretrained(sentiment_model_name)
        _sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
    return _sentiment_tokenizer, _sentiment_config, _sentiment_model

def get_absa_model():
    global _absa_model
    if _absa_model is None:
        try:
            from setfit import AbsaModel
            print("Loading ABSA models...")
            ASPECT_MODEL_ID = "tomaarsen/setfit-absa-paraphrase-mpnet-base-v2-restaurants-aspect"
            POLARITY_MODEL_ID = "tomaarsen/setfit-absa-paraphrase-mpnet-base-v2-restaurants-polarity"
            _absa_model = AbsaModel.from_pretrained(ASPECT_MODEL_ID, POLARITY_MODEL_ID)
        except ImportError:
            logging.warning("SetFit not installed. ABSA functionality will be disabled.")
            return None
        except Exception as e:
            logging.error(f"Error loading ABSA model: {e}")
            return None
    return _absa_model


# Utility: load image from URL or path
def load_image(path_or_url):
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return Image.open(requests.get(path_or_url, stream=True).raw).convert("RGB")
    elif os.path.isfile(path_or_url):
        return Image.open(path_or_url).convert("RGB")
    else:
        raise ValueError(f"Invalid image path or URL: {path_or_url}")

# Clean up text for sentiment model
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)

def get_aspect_sentiment(text: str) -> list:
    """
    Extracts aspects and their sentiment polarity from text.
    """
    if not text:
        return []
    
    model = get_absa_model()
    if model is None:
        return []

    try:
        # Predict returns a list of dicts: [{'span': 'aspect', 'polarity': 'positive'}, ...]
        return model.predict(text)
    except Exception as e:
        logging.error(f"ABSA Error: {e}")
        return []

def get_image_caption_and_sentiment(image_path_or_url: str, caption: str,  prompt: str = "a photography of"):
    raw_image = load_image(image_path_or_url)
    
    # Load models lazily
    blip_proc, blip_mod = get_blip_models()
    sent_tok, sent_conf, sent_mod = get_sentiment_models()

    # # Captioning
    # inputs = blip_proc(raw_image, prompt, return_tensors="pt")
    # with torch.no_grad():
    #     out = blip_mod.generate(**inputs)
    # conditional_img_caption = blip_proc.decode(out[0], skip_special_tokens=True)

    inputs = blip_proc(raw_image, return_tensors="pt")
    with torch.no_grad():
        out = blip_mod.generate(**inputs)
    img_caption = blip_proc.decode(out[0], skip_special_tokens=True)

    # Sentiment Analysis
    combined_text = f"{caption}"
    processed_text = preprocess(combined_text)
    encoded_input = sent_tok(processed_text, return_tensors="pt")
    with torch.no_grad():
        output = sent_mod(**encoded_input)

    scores = softmax(output.logits[0].detach().numpy())
    top_idx = np.argmax(scores)
    top_sentiment = {
        "label": sent_conf.id2label[top_idx],
        "score": float(np.round(scores[top_idx], 4))
    }

    return {
        "imgcaption": img_caption,
        "sentiment": top_sentiment  
    }

from transformers import pipeline

# Load your fine-tuned model (update path to your specific run output)
HF_MODEL_ID = "ashh007/dreams-chime-bert" 

# Initialize pipeline lazily
_chime_classifier = None

def get_chime_category(text: str):
    global _chime_classifier
    if text is None or not text.strip():
        # Handle empty text gracefully
        return {"label": "Uncategorized", "score": 0.0}

    if _chime_classifier is None:
        try:
            logging.info(f"Loading CHIME model from Hugging Face: {HF_MODEL_ID}...")
            # This line automatically downloads the model from the Hub 
            # and caches it in ~/.cache/huggingface
            _chime_classifier = pipeline(
                "text-classification", 
                model=HF_MODEL_ID, 
                tokenizer=HF_MODEL_ID,
                return_all_scores=True
            )
            print("CHIME model loaded successfully.")
        except Exception as e:
            print(f"Error loading CHIME model: {e}")
            # Fallback for when internet is down or model ID is wrong
            return {"label": "Hope", "score": 0.0}

    # Run inference
    try:
        results = _chime_classifier(text)
        # results is a list of lists of dicts: [[{'label': 'Hope', 'score': 0.9}, ...]]
        # We want the highest scoring category
        top_result = max(results[0], key=lambda x: x['score'])
        return top_result
    except Exception as e:
        print(f"Inference error: {e}")
        return {"label": "Hope", "score": 0.0}

bp = Blueprint("sentiment", __name__, url_prefix="/sentiment")


@bp.route("/analyze", methods=["POST"])
def analyze_sentiment():
    """
    Expects JSON body with:
      - image_path_or_url (str)
      - caption (str)
    Returns:
      - JSON with image caption and sentiment
    """
    data = request.get_json() or {}

    image_path_or_url = data.get("image_path_or_url")
    caption = data.get("caption", "")

    if not image_path_or_url:
        return jsonify({"error": "image_path_or_url is required"}), 400

    result = get_image_caption_and_sentiment(
        image_path_or_url=image_path_or_url,
        caption=caption,
    )
    # We analyze the user-provided caption if it exists, otherwise we could analyze the generated imgcaption
    text_to_analyze = caption if (caption and caption.strip()) else result["imgcaption"]
    result["aspect_sentiment"] = get_aspect_sentiment(text_to_analyze)
    result["chime_analysis"] = get_chime_category(text_to_analyze)

    return jsonify(result), 200
