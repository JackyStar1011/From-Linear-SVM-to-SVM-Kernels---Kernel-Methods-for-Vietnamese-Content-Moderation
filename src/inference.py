from functools import lru_cache

import joblib
import numpy as np

from src.config import DEFAULT_MODEL_KEY, LABEL_MAP, get_model_config
from src.preprocess import preprocess_text


def _normalize_label(value) -> str:
    if isinstance(value, (int, np.integer)):
        return LABEL_MAP.get(int(value), str(int(value)))

    value_str = str(value)
    if value_str.isdigit():
        return LABEL_MAP.get(int(value_str), value_str)
    return value_str


def _extract_score_map(score_source, features, classes=None) -> tuple[str | None, dict[str, float]]:
    if hasattr(score_source, "decision_function"):
        raw_scores = score_source.decision_function(features)
        score_name = "decision_score"
    elif hasattr(score_source, "predict_proba"):
        raw_scores = score_source.predict_proba(features)
        score_name = "probability"
    else:
        return None, {}

    score_array = np.asarray(raw_scores)
    if score_array.ndim == 2:
        score_array = score_array[0]
    else:
        score_array = score_array.reshape(-1)

    if classes is not None and len(classes) == len(score_array):
        score_labels = [_normalize_label(item) for item in classes]
    elif len(score_array) == len(LABEL_MAP):
        score_labels = [LABEL_MAP[idx] for idx in range(len(LABEL_MAP))]
    else:
        score_labels = [f"score_{idx}" for idx in range(len(score_array))]

    return score_name, {
        label: float(score)
        for label, score in zip(score_labels, score_array)
    }


@lru_cache(maxsize=None)
def load_artifacts(model_key: str = DEFAULT_MODEL_KEY) -> dict:
    model_config = get_model_config(model_key)
    artifact_format = model_config["artifact_format"]

    if artifact_format == "legacy_separate":
        model_path = model_config["model_path"]
        vectorizer_path = model_config["vectorizer_path"]

        if not model_path.exists():
            raise FileNotFoundError(f"model file not found: {model_path}")
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"vectorizer file not found: {vectorizer_path}")

        return {
            "artifact_format": artifact_format,
            "model_config": model_config,
            "vectorizer": joblib.load(vectorizer_path),
            "model": joblib.load(model_path),
        }

    if artifact_format == "pipeline_bundle":
        artifact_path = model_config["artifact_path"]
        if not artifact_path.exists():
            raise FileNotFoundError(f"artifact file not found: {artifact_path}")

        artifact = joblib.load(artifact_path)
        if isinstance(artifact, dict) and "pipeline" in artifact:
            return {
                "artifact_format": artifact_format,
                "model_config": model_config,
                "pipeline": artifact["pipeline"],
                "metadata": artifact.get("metadata", {}),
            }

        return {
            "artifact_format": artifact_format,
            "model_config": model_config,
            "pipeline": artifact,
            "metadata": {},
        }

    raise ValueError(f"unsupported artifact format: {artifact_format}")


def predict_comment(text: str, model_key: str = DEFAULT_MODEL_KEY) -> dict:
    loaded = load_artifacts(model_key)
    model_config = loaded["model_config"]

    processed_text = preprocess_text(text, model_config["preprocessing"])

    if loaded["artifact_format"] == "legacy_separate":
        vectorizer = loaded["vectorizer"]
        model = loaded["model"]
        features = vectorizer.transform([processed_text])
        prediction = model.predict(features)[0]
        score_name, decision_scores = _extract_score_map(
            score_source=model,
            features=features,
            classes=getattr(model, "classes_", None),
        )
    else:
        pipeline = loaded["pipeline"]
        prediction = pipeline.predict([processed_text])[0]
        classifier = pipeline.named_steps.get("classifier", pipeline)
        score_name, decision_scores = _extract_score_map(
            score_source=pipeline,
            features=[processed_text],
            classes=getattr(classifier, "classes_", None),
        )

    pred_label = _normalize_label(prediction)
    ranked_scores = sorted(
        decision_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return {
        "input_text": text,
        "processed_text": processed_text,
        "pred_label_name": pred_label,
        "pred_label_id": None,
        "score_name": score_name,
        "decision_scores": decision_scores,
        "ranked_scores": ranked_scores,
        "model_key": model_key,
        "model_display_name": model_config["display_name"],
        "model_status": model_config["status"],
        "evaluation_scope": model_config["evaluation_scope"],
        "feature_type": model_config["feature_type"],
        "notes": model_config["notes"],
    }
