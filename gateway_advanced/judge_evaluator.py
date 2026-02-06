import time
import json
import random
import logging
import os
import threading
from collections import defaultdict
from typing import Dict

import gradio as gr

# --------------------------------
# Logging
# --------------------------------
logger = logging.getLogger("judge")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
)
logger.handlers = [handler]
logger.propagate = False

# --------------------------------
# Config
# --------------------------------
WEIGHT_ARTIFACT_PATH = "/home/cdsw/model_weights.json"
EVAL_INTERVAL_SECONDS = 120
DASH_REFRESH_SECONDS = 30  # how often dashboard updates

JUDGE_MODEL = {
    "model_id": os.getenv("JUDGE_MODEL_ID"),
    "token": os.getenv("JUDGE_MODEL_TOKEN"),
    "url": os.getenv("JUDGE_MODEL_URL"),
}

# --------------------------------
# Shared state
# --------------------------------
LAST_ARTIFACT: Dict = {
    "weights": {"model-a": 0.1, "model-b": 0.1},
    "avg_scores": {"model-a": 0.0, "model-b": 0.0},
    "sample_counts": {"model-a": 0, "model-b": 0},
    "timestamp": time.time(),
}

# --------------------------------
# Mock telemetry fetch
# --------------------------------
def fetch_recent_requests():
    # TODO: Replace with actual SDK query
    return [
        {"model": "model-a", "output": "some text"},
        {"model": "model-b", "output": "another text"},
    ]


def judge_response(text: str) -> float:
    # TODO: Replace with actual judge model call
    return random.uniform(0.0, 1.0)


def compute_weights(scores: Dict[str, list[float]]) -> Dict[str, float]:
    return {model: max(0.1, sum(vals) / len(vals)) for model, vals in scores.items()}


# --------------------------------
# Judge loop (warm path)
# --------------------------------
def run_loop():
    global LAST_ARTIFACT

    while True:
        logger.info("Starting evaluation cycle")
        start_ts = time.time()

        samples = fetch_recent_requests()
        scores = defaultdict(list)

        for s in samples:
            score = judge_response(s["output"])
            scores[s["model"]].append(score)

        if scores:
            weights = compute_weights(scores)

            artifact = {
                "timestamp": start_ts,
                "window": f"last_{EVAL_INTERVAL_SECONDS}s",
                "weights": weights,
                "avg_scores": {m: sum(v) / len(v) for m, v in scores.items()},
                "sample_counts": {m: len(v) for m, v in scores.items()},
            }

            # Save artifact to disk
            try:
                with open(WEIGHT_ARTIFACT_PATH, "w") as f:
                    json.dump(artifact, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to write artifact: {e}")

            LAST_ARTIFACT = artifact
            logger.info(f"Published new weights: {weights}")
        else:
            logger.info("No samples available for evaluation")

        time.sleep(EVAL_INTERVAL_SECONDS)


# --------------------------------
# Gradio dashboard
# --------------------------------
def get_dashboard():
    import pandas as pd

    artifact = LAST_ARTIFACT
    return {
        "Metadata": f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(artifact['timestamp']))}",
        "Weights": pd.DataFrame(artifact["weights"].items(), columns=["Model", "Weight"]),
        "Avg Scores": pd.DataFrame(artifact["avg_scores"].items(), columns=["Model", "Score"]),
        "Sample Counts": pd.DataFrame(artifact["sample_counts"].items(), columns=["Model", "Samples"]),
    }


def build_gradio_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## AI Gateway — LLM-as-Judge Dashboard")

        weights_table = gr.Dataframe()
        scores_table = gr.Dataframe()
        counts_table = gr.Dataframe()
        metadata_text = gr.Markdown()

        def update():
            data = get_dashboard()
            metadata_text.update(data["Metadata"])
            weights_table.update(data["Weights"])
            scores_table.update(data["Avg Scores"])
            counts_table.update(data["Sample Counts"])

        # Refresh button
        gr.Button("Refresh").click(fn=update, inputs=[], outputs=[])

        # Auto-refresh via queue
        demo.load(fn=update, inputs=[], outputs=[], every=DASH_REFRESH_SECONDS)

    return demo


# --------------------------------
# Main launcher
# --------------------------------
if __name__ == "__main__":
    # Start judge loop in background
    threading.Thread(target=run_loop, daemon=True).start()

    # Start Gradio dashboard
    demo = build_gradio_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=int(os.environ["CDSW_APP_PORT"]),
        show_error=True,
    )
