import logging
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional in lightweight test envs
    torch = None

try:
    from scipy.special import softmax
except ImportError:  # pragma: no cover - optional in lightweight test envs
    def softmax(x):
        x = np.asarray(x)
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
    from transformers import pipeline
except ImportError:  # pragma: no cover - optional in lightweight test envs
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    AutoConfig = None
    pipeline = None

HF_MODEL_ID = "ashh007/dreams-chime-bert"

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
        self._chime_classifier = None

    def get_sentiment_models(self, sentiment_model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        if AutoTokenizer is None or AutoConfig is None or AutoModelForSequenceClassification is None:
            raise RuntimeError("transformers is required for sentiment inference")
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
        if torch is None:
            raise RuntimeError("torch is required for sentiment inference")
            
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

    def get_chime_classifier(self):
        if self._chime_classifier is None:
            try:
                if pipeline is None:
                    raise RuntimeError("transformers is required for CHIME inference")
                self._chime_classifier = pipeline(
                    "text-classification",
                    model=HF_MODEL_ID,
                    tokenizer=HF_MODEL_ID,
                    return_all_scores=True,
                )
            except Exception as e:
                logger.error(f"Error loading CHIME model: {e}")
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
            # Handle both [dict] and [[dict]] return formats from the pipeline.
            if not results:
                return {"label": "Uncategorized", "score": 0.0}

            if isinstance(results[0], list):
                if not results[0]:
                    return {"label": "Uncategorized", "score": 0.0}
                return max(results[0], key=lambda x: x.get("score", 0.0))

            if isinstance(results[0], dict):
                return max(results, key=lambda x: x.get("score", 0.0))

            return {"label": "Uncategorized", "score": 0.0}
        except Exception as e:
            logger.error(f"CHIME inference error: {e}")
            return {"label": "Uncategorized", "score": 0.0}

analyzer = SentimentAnalyzer()

def get_sentiment(text: str, sentiment_model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
    return analyzer.get_sentiment(text, sentiment_model_name)


def get_chime_category(text: str):
    return analyzer.analyze_chime(text)
