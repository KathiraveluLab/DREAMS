import os
import logging
import torch
from transformers import pipeline

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
                model_path = HF_MODEL_ID

                # Try to detect Flask context safely
                try:
                    from flask import has_app_context, current_app

                    if has_app_context():
                        local_model_path = os.path.join(
                            current_app.root_path,
                            "models",
                            "production_chime_model"
                        )

                        if os.path.exists(local_model_path):
                            logging.info(
                                f">>> SELF-CORRECTION: Learned model found at {local_model_path}. Loading..."
                            )
                            model_path = local_model_path
                        else:
                            logging.info(
                                f"Loading Base CHIME model from Hugging Face: {HF_MODEL_ID}"
                            )

                    else:
                        logging.info(
                            "No Flask context detected. Using default HuggingFace model."
                        )

                except RuntimeError:
                    # pytest will land here
                    logging.info(
                        "Running outside Flask. Using default HuggingFace model."
                    )

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
            
            # HuggingFace pipelines can return [dict] or [[dict]] depending on configuration
            if len(results) > 0 and isinstance(results[0], list):
                top_result = max(results[0], key=lambda x: x.get('score', 0))
            else:
                top_result = max(results, key=lambda x: x.get('score', 0))
                
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


# Singleton instance for general use
adv_analyzer = AdvancedSentimentAnalyzer()

# Maintaining functional interface for backward compatibility
def get_chime_category(text: str):
    return adv_analyzer.analyze_chime(text)

def get_aspect_sentiment(text: str):
    return adv_analyzer.analyze_aspect_sentiment(text)
