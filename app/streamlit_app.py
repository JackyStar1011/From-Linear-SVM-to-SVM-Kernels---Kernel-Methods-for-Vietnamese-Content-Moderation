from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import MAX_INPUT_CHARS, MODEL_REGISTRY, get_model_config, get_model_keys_by_family
from src.inference import predict_comment


FAMILY_LABELS = {
    "linear": "Linear SVM",
    "kernel": "Kernel SVM",
}

LABEL_STYLES = {
    "CLEAN": {
        "background": "#e8f7ee",
        "border": "#16a34a",
        "text": "#166534",
    },
    "OFFENSIVE": {
        "background": "#fff7db",
        "border": "#f59e0b",
        "text": "#92400e",
    },
    "HATE": {
        "background": "#fdeaea",
        "border": "#dc2626",
        "text": "#991b1b",
    },
}


def apply_page_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 880px;
            padding-top: 2rem;
            padding-bottom: 2.5rem;
        }
        .stTextArea textarea {
            min-height: 220px;
            font-size: 1rem;
            line-height: 1.5;
        }
        .stButton > button {
            width: 100%;
            min-height: 3rem;
            font-size: 1rem;
            font-weight: 700;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_model_selection() -> str:
    st.sidebar.header("Config")

    selected_family = st.sidebar.radio(
        "Model family",
        options=["linear", "kernel"],
        format_func=lambda family: FAMILY_LABELS[family],
        index=0,
    )

    if selected_family == "linear":
        return "linear"

    kernel_keys = get_model_keys_by_family("kernel")
    return st.sidebar.selectbox(
        "Kernel type",
        options=kernel_keys,
        format_func=lambda key: MODEL_REGISTRY[key]["display_name"],
        index=0,
    )


def render_result_banner(result: dict) -> None:
    pred_label = result["pred_label_name"]
    style = LABEL_STYLES.get(pred_label, LABEL_STYLES["CLEAN"])

    st.markdown(
        f"""
        <div style="
            background:{style['background']};
            border:1px solid {style['border']};
            border-left:8px solid {style['border']};
            border-radius:8px;
            padding:18px 20px;
            margin: 8px 0 14px 0;
        ">
            <div style="font-size:0.85rem;color:{style['text']};font-weight:700;margin-bottom:6px;">
                Predicted label
            </div>
            <div style="font-size:2rem;color:{style['text']};font-weight:800;line-height:1.1;">
                {pred_label}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_score_bars(result: dict) -> None:
    ranked_scores = result["ranked_scores"]
    if not ranked_scores:
        st.info("This model does not expose decision scores.")
        return

    values = [score for _, score in ranked_scores]
    low = min(values)
    high = max(values)
    score_name = result["score_name"] or "score"

    st.markdown("#### Model scores")
    for label, score in ranked_scores:
        if high == low:
            normalized = 1.0
        else:
            normalized = (score - low) / (high - low)

        left, right = st.columns([3, 1])
        with left:
            st.write(label)
            st.progress(float(normalized))
        with right:
            st.write(f"{score:.4f}")

    st.caption(f"Displayed as normalized bars based on `{score_name}`.")


def render_model_info(model_key: str) -> None:
    cfg = get_model_config(model_key)
    st.markdown("#### Model details")
    st.write(f"Selected model: `{cfg['display_name']}`")
    st.write(f"Status: `{cfg['status']}`")
    st.write(f"Evaluation scope: `{cfg['evaluation_scope']}`")
    st.write(f"Preprocessing: `{cfg['preprocessing']}`")
    st.write(f"Feature type: `{cfg['feature_type']}`")
    if cfg["artifact_format"] == "legacy_separate":
        st.write(f"Model path: `{cfg['model_path']}`")
        st.write(f"Vectorizer path: `{cfg['vectorizer_path']}`")
    else:
        st.write(f"Artifact path: `{cfg['artifact_path']}`")
    if cfg["notes"]:
        st.caption(cfg["notes"])


st.set_page_config(
    page_title="Vietnamese Moderation Tool",
    layout="centered",
)

apply_page_styles()

selected_model_key = build_model_selection()
selected_model = get_model_config(selected_model_key)

st.title("Vietnamese Content Moderation")
st.caption("Enter one Vietnamese sentence and classify it into CLEAN, OFFENSIVE, or HATE.")

if selected_model["evaluation_scope"] != "full_test":
    st.info(
        f"{selected_model['display_name']} is currently loaded from a `{selected_model['evaluation_scope']}` artifact."
    )

comment = st.text_area(
    label="Input comment",
    placeholder="Type a Vietnamese sentence here...",
    max_chars=MAX_INPUT_CHARS,
)
st.caption(f"{len(comment)}/{MAX_INPUT_CHARS} characters")

predict_clicked = st.button("Predict label", type="primary")

st.divider()

if predict_clicked:
    if not comment.strip():
        st.warning("Please enter a comment before predicting.")
    else:
        try:
            result = predict_comment(comment, model_key=selected_model_key)
            render_result_banner(result)
            render_score_bars(result)

            with st.expander("Processed text", expanded=False):
                st.code(result["processed_text"])

            with st.expander("Model info", expanded=False):
                render_model_info(selected_model_key)

        except Exception as exc:
            st.error(f"{selected_model['display_name']} failed: {exc}")
else:
    with st.expander("Model info", expanded=False):
        render_model_info(selected_model_key)
