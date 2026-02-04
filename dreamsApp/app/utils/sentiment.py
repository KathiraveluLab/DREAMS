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
from transformers import pipeline

HF_MODEL_ID = "ashh007/dreams-chime-bert"

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

def select_text_for_analysis(caption: str, generated_caption: str) -> str:
    """DRY: Utility to prioritize user caption over auto-generated one."""
    return caption if (caption and caption.strip()) else generated_caption

class SentimentAnalyzer:
    def __init__(self):
        self._blip_processor = None
        self._blip_model = None
        self._sentiment_tokenizer = None
        self._sentiment_config = None
        self._sentiment_model = None
        self._absa_model = None
        self._chime_classifier = None

    def get_blip_models(self):
        if self._blip_processor is None or self._blip_model is None:
            print("Loading Blip models...")
            self._blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            self._blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        return self._blip_processor, self._blip_model

    def get_sentiment_models(self):
        if self._sentiment_model is None:
            print("Loading Sentiment models...")
            sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            self._sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
            self._sentiment_config = AutoConfig.from_pretrained(sentiment_model_name)
            self._sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
        return self._sentiment_tokenizer, self._sentiment_config, self._sentiment_model

    def get_absa_model(self):
        if self._absa_model is None:
            try:
                from setfit import AbsaModel
                print("Loading ABSA models...")
                ASPECT_MODEL_ID = "tomaarsen/setfit-absa-paraphrase-mpnet-base-v2-restaurants-aspect"
                POLARITY_MODEL_ID = "tomaarsen/setfit-absa-paraphrase-mpnet-base-v2-restaurants-polarity"
                self._absa_model = AbsaModel.from_pretrained(ASPECT_MODEL_ID, POLARITY_MODEL_ID)
            except ImportError:
                logging.warning("SetFit not installed. ABSA functionality will be disabled.")
                return None
            except Exception as e:
                logging.error(f"Error loading ABSA model: {e}")
                return None
        return self._absa_model

    def get_chime_classifier(self):
        if self._chime_classifier is None:
            try:
                # Check for locally fine-tuned model (Self-Correcting Feature)
                # sentiment.py is in dreamsApp/app/utils, so ../../models is the path
                from flask import current_app
                local_model_path = os.path.join(current_app.root_path, "models", "production_chime_model")
                
                model_path = HF_MODEL_ID
                
                if os.path.exists(local_model_path):
                    logging.info(f">>> SELF-CORRECTION: Learned model found at {local_model_path}. Loading...")
                    model_path = local_model_path
                else:
                    logging.info(f"Loading Base CHIME model from Hugging Face: {HF_MODEL_ID}...")

                self._chime_classifier = pipeline(
                    "text-classification", 
                    model=model_path, 
                    tokenizer=model_path,
                    return_all_scores=True
                )
                logging.info("CHIME model loaded successfully.")
            except Exception as e:
                logging.error(f"Error loading CHIME model: {e}")
                return None
        return self._chime_classifier

    def analyze_chime(self, text: str):
        if text is None or not text.strip():
            return {"label": "Uncategorized", "score": 0.0}

        classifier = self.get_chime_classifier()
        if classifier is None:
            return {"label": "Uncategorized", "score": 0.0}

        try:
            results = classifier(text)
            top_result = max(results[0], key=lambda x: x['score'])
            return top_result
        except Exception as e:
            print(f"Inference error: {e}")
            return {"label": "Uncategorized", "score": 0.0}

    def analyze_aspect_sentiment(self, text: str) -> list:
        if not text:
            return []
        
        model = self.get_absa_model()
        if model is None:
            return []

        try:
            return model.predict(text)
        except Exception as e:
            logging.error(f"ABSA Error: {e}")
            return []

    def get_image_caption_and_sentiment(self, image_path_or_url: str, caption: str, prompt: str = "a photography of"):
        raw_image = load_image(image_path_or_url)
        
        blip_proc, blip_mod = self.get_blip_models()
        sent_tok, sent_conf, sent_mod = self.get_sentiment_models()

        # Note: prompt argument currently unused in current implementation 
        # but kept for API consistency
        inputs = blip_proc(raw_image, return_tensors="pt")
        with torch.no_grad():
            out = blip_mod.generate(**inputs)
        img_caption = blip_proc.decode(out[0], skip_special_tokens=True)

        processed_text = preprocess(caption) if caption else ""
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

# Singleton instance for general use
analyzer = SentimentAnalyzer()

# Maintaining functional interface for backward compatibility
def get_chime_category(text: str):
    return analyzer.analyze_chime(text)

def get_aspect_sentiment(text: str):
    return analyzer.analyze_aspect_sentiment(text)

def get_image_caption_and_sentiment(image_path_or_url: str, caption: str, prompt: str = "a photography of"):
    return analyzer.get_image_caption_and_sentiment(image_path_or_url, caption, prompt)


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
    # Refactor: Use shared selection logic
    text_to_analyze = select_text_for_analysis(caption, result["imgcaption"])
    result["aspect_sentiment"] = get_aspect_sentiment(text_to_analyze)
    result["chime_analysis"] = get_chime_category(text_to_analyze)

    return jsonify(result), 200
