# reward/scorer.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "rm_model") 


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

model.eval()

# Inference function
def score_answer(text: str) -> float:
    """Returns a score between 0 (bad) and 1 (good) for the given answer."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        good_score = probs[0][1].item()  # Index 1 = 'good' class
    return good_score
