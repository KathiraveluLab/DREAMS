import logging
from dreams_app.core.sentiment import get_chime_category as core_get_chime_category

try:
    import torch
except ImportError:  # pragma: no cover - optional in lightweight test envs
    torch = None

class AdvancedSentimentAnalyzer:
    """Handles heavy experimental sentiment models like ABSA."""
    def __init__(self):
        self._absa_model = None

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
    return core_get_chime_category(text)

def get_aspect_sentiment(text: str):
    return adv_analyzer.analyze_aspect_sentiment(text)
