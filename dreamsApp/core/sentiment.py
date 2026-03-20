import logging
import numpy as np
import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

logger = logging.getLogger(__name__)

def preprocess(text):
    if not text:
        return ""
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)

class SentimentAnalyzer:
    def __init__(self):
        self._sentiment_tokenizer = None
        self._sentiment_config = None
        self._sentiment_model = None
        self._sentiment_model_name = None

    def get_sentiment_models(self, sentiment_model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        if self._sentiment_model is None or self._sentiment_model_name != sentiment_model_name:
            logger.info(f"Loading Sentiment model ({sentiment_model_name})...")
            self._sentiment_model_name = sentiment_model_name
            self._sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
            self._sentiment_config = AutoConfig.from_pretrained(sentiment_model_name)
            self._sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
        return self._sentiment_tokenizer, self._sentiment_config, self._sentiment_model

    def get_sentiment(self, text: str, sentiment_model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        if not text or not text.strip():
            return {"label": "neutral", "score": 1.0}
            
        sent_tok, sent_conf, sent_mod = self.get_sentiment_models(sentiment_model_name)
        processed_text = preprocess(text)
        encoded_input = sent_tok(processed_text, return_tensors="pt")
        with torch.no_grad():
            output = sent_mod(**encoded_input)

        scores = softmax(output.logits[0].detach().numpy())
        top_idx = np.argmax(scores)
        top_sentiment = {
            "label": sent_conf.id2label[top_idx],
            "score": float(np.round(scores[top_idx], 4))
        }
        return top_sentiment

analyzer = SentimentAnalyzer()

def get_sentiment(text: str, sentiment_model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
    return analyzer.get_sentiment(text, sentiment_model_name)
