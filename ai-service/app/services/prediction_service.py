# ai-service/app/services/prediction_service.py
import os
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from app.core.config import settings

class PredictionService:
    def __init__(self):
        model_dir = os.path.join(settings.MODEL_DIR, settings.FINE_TUNED_MODEL_NAME)
        opt_model_path = os.path.join(model_dir, settings.OPTIMIZED_MODEL_NAME)

        # Load tokenizer and model
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        
        # Load quantized weights
        state_dict = torch.load(opt_model_path, map_location="cpu")
        self.model.load_state_dict(state_dict)

        self.model.eval()

    def predict(self, text: str) -> dict:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        preds = torch.argmax(logits, dim=-1).item()
        return {"predicted_priority": preds}
