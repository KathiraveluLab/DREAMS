import os
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import requests

# Load models once
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_config = AutoConfig.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

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

# Main function
def get_image_caption_and_sentiment(image_path_or_url: str, caption: str,  prompt: str = "a photography of"):
    raw_image = load_image(image_path_or_url)

    # # Captioning
    # inputs = blip_processor(raw_image, prompt, return_tensors="pt")
    # with torch.no_grad():
    #     out = blip_model.generate(**inputs)
    # conditional_img_caption = blip_processor.decode(out[0], skip_special_tokens=True)

    inputs = blip_processor(raw_image, return_tensors="pt")
    with torch.no_grad():
        out = blip_model.generate(**inputs)
    img_caption = blip_processor.decode(out[0], skip_special_tokens=True)

    # Sentiment Analysis
    combined_text = f"{caption} {img_caption}"
    processed_text = preprocess(combined_text)
    encoded_input = sentiment_tokenizer(processed_text, return_tensors="pt")
    with torch.no_grad():
        output = sentiment_model(**encoded_input)

    scores = softmax(output.logits[0].detach().numpy())
    top_idx = np.argmax(scores)
    top_sentiment = {
        "label": sentiment_config.id2label[top_idx],
        "score": float(np.round(scores[top_idx], 4))
    }

    return {
        "imgcaption": img_caption,
        "sentiment": top_sentiment  
    }
