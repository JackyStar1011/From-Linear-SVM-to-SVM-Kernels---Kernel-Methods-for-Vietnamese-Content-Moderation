from functools import lru_cache

import joblib
import numpy as np

from src.config import LABEL_MAP, MODEL_PATH, PREPROCESSING_NAME, VECTORIZER_PATH
from src.preprocess import preprocess_text


@lru_cache(maxsize=1)
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"model file not found: {MODEL_PATH}")
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"vectorizer file not found: {VECTORIZER_PATH}")

    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    return vectorizer, model


def predict_comment(text: str) -> dict:
    vectorizer, model = load_artifacts()

    processed_text = preprocess_text(text, PREPROCESSING_NAME)
    X = vectorizer.transform([processed_text])

    pred_id = int(model.predict(X)[0])
    pred_label = LABEL_MAP[pred_id]

    raw_scores = model.decision_function(X)
    raw_scores = np.asarray(raw_scores).reshape(-1)

    decision_scores = {
        LABEL_MAP[i]: float(raw_scores[i]) for i in range(len(raw_scores))
    }

    ranked_scores = sorted(
        decision_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return {
        "input_text": text,
        "processed_text": processed_text,
        "pred_label_id": pred_id,
        "pred_label_name": pred_label,
        "decision_scores": decision_scores,
        "ranked_scores": ranked_scores,
    }
