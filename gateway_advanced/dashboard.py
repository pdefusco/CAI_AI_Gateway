import streamlit as st
import sqlite3
import pandas as pd
import time

DB_PATH = "/tmp/requests.db"
REFRESH_SECONDS = 10

st.set_page_config(
    page_title="AI Gateway Dashboard",
    layout="wide"
)

st.title("AI Gateway — Model Routing Dashboard")

# -----------------------------
# Auto-refresh
# -----------------------------
from streamlit_autorefresh import st_autorefresh

st.caption(f"Auto-refresh every {REFRESH_SECONDS}s")
count = st_autorefresh(interval=REFRESH_SECONDS * 1000, limit=None, key="dashboard_autorefresh")

# -----------------------------
# Connect to SQLite
# -----------------------------
conn = sqlite3.connect(DB_PATH, timeout=60, check_same_thread=False)
c = conn.cursor()

# -----------------------------
# Load latest model weights
# -----------------------------
c.execute("""
    SELECT model, weight, last_updated
    FROM model_weights
    WHERE last_updated = (SELECT MAX(last_updated) FROM model_weights)
""")
rows = c.fetchall()

if not rows:
    st.warning("No model weights found in the database yet")
    st.stop()

weights_df = pd.DataFrame(rows, columns=["model", "weight", "last_updated"])
weights_df = weights_df.drop(columns="last_updated")  # Optional, just for display

# -----------------------------
# Model weights chart
# -----------------------------
st.subheader("Routing Weights (Latest)")
st.bar_chart(weights_df.set_index("model"))

# -----------------------------
# Optional: show historical weights over time
# -----------------------------
st.subheader("Routing Weights History")
c.execute("""
    SELECT last_updated, model, weight
    FROM model_weights_history
    ORDER BY last_updated ASC
""")

history_rows = c.fetchall()
if history_rows:
    history_df = pd.DataFrame(history_rows, columns=["last_updated", "model", "weight"])
    history_df["last_updated"] = pd.to_datetime(history_df["last_updated"], unit='s')
    st.line_chart(history_df.pivot(index="last_updated", columns="model", values="weight"))

# -----------------------------
# Design note
# -----------------------------
st.markdown(
    """
    This Dashboard shows updated model weights in near real time.
    Model weights are assigned by the LLM Judge after reviewing model responses to the provided inputs.
    The gateway periodically reads these model weights and routes incoming requests to the preferred model.
    """
)
