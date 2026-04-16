from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT_DIR / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "final_linear_svm_model.joblib"
VECTORIZER_PATH = ARTIFACT_DIR / "final_linear_svm_vectorizer.joblib"

TEXT_COL = "free_text"
LABEL_COL = "label_id"

LABEL_MAP = {
    0: "CLEAN",
    1: "OFFENSIVE",
    2: "HATE",
}

MAX_INPUT_CHARS = 300
PREPROCESSING_NAME = "strong"
