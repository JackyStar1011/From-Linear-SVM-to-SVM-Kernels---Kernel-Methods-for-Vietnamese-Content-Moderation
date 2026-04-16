from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT_DIR / "artifacts"

TEXT_COL = "free_text"
LABEL_COL = "label_id"

LABEL_MAP = {
    0: "CLEAN",
    1: "OFFENSIVE",
    2: "HATE",
}
LABEL_NAME_TO_ID = {label: idx for idx, label in LABEL_MAP.items()}

MAX_INPUT_CHARS = 300
DEFAULT_MODEL_KEY = "linear"

MODEL_REGISTRY = {
    "linear": {
        "display_name": "Linear SVM",
        "family": "linear",
        "artifact_format": "legacy_separate",
        "model_path": ARTIFACT_DIR / "final_linear_svm_model.joblib",
        "vectorizer_path": ARTIFACT_DIR / "final_linear_svm_vectorizer.joblib",
        "preprocessing": "strong",
        "feature_type": "char_wb (3,5)",
        "status": "stable",
        "evaluation_scope": "full_test",
        "notes": "Baseline model from the partner repo, evaluated on the full ViHSD test split.",
    },
    "poly": {
        "display_name": "Polynomial Kernel SVM",
        "family": "kernel",
        "kernel_name": "poly",
        "artifact_format": "pipeline_bundle",
        "artifact_path": ARTIFACT_DIR / "poly_svm_smoketest_word.joblib",
        "preprocessing": "kernel_default",
        "feature_type": "word (1,2)",
        "status": "experimental",
        "evaluation_scope": "smoke_test",
        "notes": "Kernel artifact imported from the local research pipeline. Current weights come from a smoke-test run.",
    },
    "rbf": {
        "display_name": "RBF Kernel SVM",
        "family": "kernel",
        "kernel_name": "rbf",
        "artifact_format": "pipeline_bundle",
        "artifact_path": ARTIFACT_DIR / "rbf_svm_smoketest_word.joblib",
        "preprocessing": "kernel_default",
        "feature_type": "word (1,2)",
        "status": "experimental",
        "evaluation_scope": "smoke_test",
        "notes": "Kernel artifact imported from the local research pipeline. Current weights come from a smoke-test run.",
    },
}


def get_model_config(model_key: str) -> dict:
    if model_key not in MODEL_REGISTRY:
        valid_keys = ", ".join(MODEL_REGISTRY)
        raise ValueError(f"unknown model key: {model_key}. expected one of: {valid_keys}")
    return MODEL_REGISTRY[model_key]


def get_model_keys_by_family(family: str) -> list[str]:
    return [
        model_key
        for model_key, model_config in MODEL_REGISTRY.items()
        if model_config.get("family") == family
    ]


MODEL_PATH = MODEL_REGISTRY[DEFAULT_MODEL_KEY]["model_path"]
VECTORIZER_PATH = MODEL_REGISTRY[DEFAULT_MODEL_KEY]["vectorizer_path"]
PREPROCESSING_NAME = MODEL_REGISTRY[DEFAULT_MODEL_KEY]["preprocessing"]
