import time
import json
import random
import logging
import requests
import os
from collections import defaultdict

logger = logging.getLogger("judge")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.handlers = [handler]

WEIGHT_ARTIFACT_PATH = "/tmp/model_weights.json"
EVAL_INTERVAL_SECONDS = 120

JUDGE_MODEL = {
    "model_id": os.getenv("JUDGE_MODEL_ID"),
    "token": os.getenv("JUDGE_MODEL_TOKEN"),
    "url": os.getenv("JUDGE_MODEL_URL"),
}

# -----------------------------
# Mock telemetry fetch
# -----------------------------
def fetch_recent_requests():
    # Replace with SDK query
    return [
        {"model": "model-a", "output": "some text"},
        {"model": "model-b", "output": "another text"},
    ]


def judge_response(text: str) -> float:
    # Replace with real judge prompt
    # For now: fake score
    return random.uniform(0.0, 1.0)


def compute_weights(scores):
    weights = {}
    for model, vals in scores.items():
        weights[model] = max(0.1, sum(vals) / len(vals))
    return weights


def run_loop():
    while True:
        logger.info("Starting evaluation cycle")

        samples = fetch_recent_requests()
        scores = defaultdict(list)

        for s in samples:
            score = judge_response(s["output"])
            scores[s["model"]].append(score)

        if scores:
            weights = compute_weights(scores)
            artifact = {
                "timestamp": time.time(),
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

            logger.info(f"Published new weights: {weights}")

        time.sleep(EVAL_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_loop()
