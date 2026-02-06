import streamlit as st
import json
import os
import time
import pandas as pd

WEIGHT_ARTIFACT_PATH = "/tmp/model_weights.json"
REFRESH_SECONDS = 30

st.set_page_config(
    page_title="AI Gateway Dashboard",
    layout="wide"
)

st.title("AI Gateway — Model Routing Dashboard")

# -----------------------------
# Auto-refresh
# -----------------------------
st.caption(f"Auto-refresh every {REFRESH_SECONDS}s")
time.sleep(REFRESH_SECONDS)
st.experimental_rerun()

# -----------------------------
# Load weight artifact
# -----------------------------
if not os.path.exists(WEIGHT_ARTIFACT_PATH):
    st.warning("No weight artifact found yet")
    st.stop()

with open(WEIGHT_ARTIFACT_PATH) as f:
    artifact = json.load(f)

# -----------------------------
# Metadata
# -----------------------------
st.subheader("Artifact Metadata")
st.json({
    "timestamp": artifact.get("timestamp"),
    "version": artifact.get("version", "n/a"),
})

# -----------------------------
# Model weights
# -----------------------------
st.subheader("Routing Weights")

weights_df = pd.DataFrame(
    artifact["weights"].items(),
    columns=["model", "weight"]
)

st.bar_chart(weights_df.set_index("model"))

# -----------------------------
# Quality metrics
# -----------------------------
if "avg_scores" in artifact:
    st.subheader("LLM-as-Judge — Avg Score")

    scores_df = pd.DataFrame(
        artifact["avg_scores"].items(),
        columns=["model", "avg_score"]
    )

    st.bar_chart(scores_df.set_index("model"))

# -----------------------------
# Sampling coverage
# -----------------------------
if "sample_counts" in artifact:
    st.subheader("Evaluation Coverage")

    counts_df = pd.DataFrame(
        artifact["sample_counts"].items(),
        columns=["model", "samples"]
    )

    st.table(counts_df)

# -----------------------------
# Design note
# -----------------------------
st.markdown(
    """
    **Notes**
    - This dashboard is read-only
    - Data comes from derived artifacts + SDK telemetry
    - No routing or evaluation logic runs here
    """
)
