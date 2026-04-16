from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import LABEL_MAP, MAX_INPUT_CHARS, MODEL_PATH, PREPROCESSING_NAME, VECTORIZER_PATH
from src.inference import predict_comment


st.set_page_config(
    page_title="ViHSD Linear SVM Demo",
    page_icon="🛡️",
    layout="centered",
)

st.title("Vietnamese Content Moderation Demo")
st.caption("final Linear SVM inference app for CLEAN / OFFENSIVE / HATE")

with st.container():
    st.markdown("### input comment")
    comment = st.text_area(
        label="comment",
        label_visibility="collapsed",
        placeholder="type a Vietnamese comment here...",
        height=180,
        max_chars=MAX_INPUT_CHARS,
    )
    st.caption(f"{len(comment)}/{MAX_INPUT_CHARS} characters")

    predict_clicked = st.button("Predict label", use_container_width=True)

st.divider()

with st.expander("model info", expanded=False):
    st.write(f"model path: `{MODEL_PATH}`")
    st.write(f"vectorizer path: `{VECTORIZER_PATH}`")
    st.write(f"preprocessing: `{PREPROCESSING_NAME}`")
    st.write(f"labels: `{LABEL_MAP}`")

if predict_clicked:
    if not comment.strip():
        st.warning("please enter a comment before predicting.")
    else:
        try:
            result = predict_comment(comment)

            st.markdown("### prediction")
            pred_label = result["pred_label_name"]

            if pred_label == "CLEAN":
                st.success(f"Predicted label: {pred_label}")
            elif pred_label == "OFFENSIVE":
                st.warning(f"Predicted label: {pred_label}")
            else:
                st.error(f"Predicted label: {pred_label}")

            st.markdown("### decision scores")
            score_df = pd.DataFrame(
                result["ranked_scores"],
                columns=["label", "decision_score"],
            )
            st.dataframe(score_df, use_container_width=True, hide_index=True)

            st.markdown("### processed text")
            st.code(result["processed_text"])

        except Exception as exc:
            st.error(f"inference failed: {exc}")
