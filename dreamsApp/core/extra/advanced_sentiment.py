import logging
from dreamsApp.core.chime_classifier import init_chime_classifier, pick_top_chime_result

try:
    import torch
except ImportError:  # pragma: no cover - optional in lightweight test envs
    torch = None

try:
    from transformers import pipeline
except ImportError:  # pragma: no cover - optional in lightweight test envs
    pipeline = None

HF_MODEL_ID = "ashh007/dreams-chime-bert"

class AdvancedSentimentAnalyzer:
    """Handles heavy experimental sentiment models like ABSA and CHIME."""
    def __init__(self):
        self._absa_model = None
        self._chime_classifier = None

    def get_absa_model(self):
        if self._absa_model is None:
            try:
                from setfit import AbsaModel
                logging.info("Loading ABSA models...")
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
        self._chime_classifier = init_chime_classifier(
            self._chime_classifier,
            pipeline,
            HF_MODEL_ID,
            logging.getLogger(__name__),
        )
        return self._chime_classifier

    def analyze_chime(self, text: str):
        if text is None or not text.strip():
            return {"label": "Uncategorized", "score": 0.0}

        classifier = self.get_chime_classifier()
        if classifier is None:
            return {"label": "Uncategorized", "score": 0.0}

        try:
            results = classifier(text)
            return pick_top_chime_result(results)
        except Exception as e:
            logging.error(f"Inference error: {e}")
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


# Singleton instance for general use
adv_analyzer = AdvancedSentimentAnalyzer()

# Maintaining functional interface for backward compatibility
def get_chime_category(text: str):
    return adv_analyzer.analyze_chime(text)

def get_aspect_sentiment(text: str):
    return adv_analyzer.analyze_aspect_sentiment(text)
