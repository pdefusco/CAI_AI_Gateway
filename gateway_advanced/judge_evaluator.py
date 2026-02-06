import time
import json
import random
import logging
import requests
import os
import threading
from collections import defaultdict
from typing import Dict

from fastapi import FastAPI
import uvicorn

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

JUDGE_MODEL = {
    "model_id": os.getenv("JUDGE_MODEL_ID"),
    "token": os.getenv("JUDGE_MODEL_TOKEN"),
    "url": os.getenv("JUDGE_MODEL_URL"),
}

# --------------------------------
# FastAPI app (read-only)
# --------------------------------
app = FastAPI()

LAST_RUN_TS: float | None = None
LAST_ARTIFACT: Dict | None = None

# --------------------------------
# Mock telemetry fetch
# --------------------------------
def fetch_recent_requests():
    # Replace with SDK query
    return [
        {"model": "model-a", "output": "some text"},
        {"model": "model-b", "output": "another text"},
    ]


def judge_response(text: str) -> float:
    # Replace with real judge prompt + model call
    return random.uniform(0.0, 1.0)


def compute_weights(scores: Dict[str, list[float]]) -> Dict[str, float]:
    return {
        model: max(0.1, sum(vals) / len(vals))
        for model, vals in scores.items()
    }


# --------------------------------
# Judge loop (control plane owner)
# --------------------------------
def run_loop():
    global LAST_RUN_TS, LAST_ARTIFACT

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
                "avg_scores": {
                    m: sum(v) / len(v) for m, v in scores.items()
                },
                "sample_counts": {
                    m: len(v) for m, v in scores.items()
                },
            }

            with open(WEIGHT_ARTIFACT_PATH, "w") as f:
                json.dump(artifact, f, indent=2)

            LAST_RUN_TS = start_ts
            LAST_ARTIFACT = artifact

            logger.info(f"Published new weights: {weights}")
        else:
            logger.info("No samples available for evaluation")

        time.sleep(EVAL_INTERVAL_SECONDS)


# --------------------------------
# API endpoints (observability only)
# --------------------------------
@app.get("/ping")
def ping():
    return {"ok": True}


@app.get("/status")
def status():
    return {
        "last_run_ts": LAST_RUN_TS,
        "artifact_path": WEIGHT_ARTIFACT_PATH,
        "eval_interval_seconds": EVAL_INTERVAL_SECONDS,
    }


@app.get("/artifact")
def artifact():
    if LAST_ARTIFACT:
        return LAST_ARTIFACT

    if not os.path.exists(WEIGHT_ARTIFACT_PATH):
        return {"status": "no artifact yet"}

    with open(WEIGHT_ARTIFACT_PATH) as f:
        return json.load(f)


# --------------------------------
# Server launcher (Cloudera AI style)
# --------------------------------
def run_server():
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.environ["CDSW_APP_PORT"]),
        log_level="warning",
    )


if __name__ == "__main__":
    # Start judge loop (warm path)
    threading.Thread(target=run_loop, daemon=True).start()

    # Start API server in its own thread (CDSW-safe)
    threading.Thread(target=run_server, daemon=True).start()

    # Keep the main thread alive
    while True:
        time.sleep(60)
