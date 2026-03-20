import os
import logging
import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

logger = logging.getLogger(__name__)

def load_image(path_or_url):
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return Image.open(requests.get(path_or_url, stream=True).raw).convert("RGB")
    elif os.path.isfile(path_or_url):
        return Image.open(path_or_url).convert("RGB")
    else:
        raise ValueError(f"Invalid image path or URL: {path_or_url}")

class ImageCaptioner:
    """Extracts machine-generated captions using BLIP."""
    def __init__(self):
        self._blip_processor = None
        self._blip_model = None

    def get_blip_models(self):
        if self._blip_processor is None or self._blip_model is None:
            logger.info("Loading Blip models...")
            self._blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            self._blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        return self._blip_processor, self._blip_model

    def get_image_caption(self, image_path_or_url: str):
        raw_image = load_image(image_path_or_url)
        blip_proc, blip_mod = self.get_blip_models()
        inputs = blip_proc(raw_image, return_tensors="pt")
        with torch.no_grad():
            out = blip_mod.generate(**inputs)
        return blip_proc.decode(out[0], skip_special_tokens=True)

captioner = ImageCaptioner()

def get_image_caption(image_path_or_url: str):
    return captioner.get_image_caption(image_path_or_url)
