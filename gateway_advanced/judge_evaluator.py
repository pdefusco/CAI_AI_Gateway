import time
import random
import logging
import requests
import os
import threading
import sqlite3
from collections import defaultdict
from typing import Dict
from fastapi import FastAPI
import uvicorn
import re

# --------------------------------
# Logging
# --------------------------------
logger = logging.getLogger("judge")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.handlers = [handler]
logger.propagate = False

# --------------------------------
# Config
# --------------------------------
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
LAST_WEIGHTS: Dict[str, float] | None = None

# --------------------------------
# SQLite helper
# --------------------------------
DB_PATH = "requests.db"

def get_conn():
    """Return a new SQLite connection with timeout for concurrency"""
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")  # Enable WAL mode for concurrent reads/writes
    return conn

# Initialize tables if they don't exist
with get_conn() as conn:
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id TEXT,
        model TEXT,
        score REAL,
        timestamp REAL
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS model_weights (
        model TEXT PRIMARY KEY,
        weight REAL NOT NULL,
        last_updated REAL DEFAULT (strftime('%s','now'))
    )
    """)
    # Initialize weights if empty
    for model in ["model-a", "model-b"]:
        c.execute(
            "INSERT OR IGNORE INTO model_weights (model, weight) VALUES (?, ?)",
            (model, 1.0)
        )
    conn.commit()

# --------------------------------
# Fetch requests from SQLite
# --------------------------------
def fetch_recent_requests():
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT request_id, model_chosen, user_input, model_output FROM requests WHERE model_output IS NOT NULL")
        rows = c.fetchall()
    samples = []
    for r in rows:
        samples.append({
            "request_id": r[0],
            "model": r[1],
            "user_input": r[2],
            "output": r[3]
        })
    return samples

# --------------------------------
# Judge scoring
# --------------------------------
def extract_score(score_text: str) -> float:
    match = re.findall(r"([0-1](?:\.\d+)?)", score_text)
    if match:
        score = float(match[-1])
        return min(max(score, 0.0), 1.0)
    else:
        logger.warning(f"No numeric score found in judge response. Falling back to random score.")
        return random.uniform(0.0, 1.0)

def judge_response(user_input: str, model_output: str) -> float:
    prompt = (
        "You are a judge. Given the following question and answer, "
        "rate the quality, relevance, and correctness of the answer on a scale from 0 to 1. "
        "Respond with only a number.\n\n"
        f"Question: {user_input}\n"
        f"Answer: {model_output}"
    )

    payload = {
        "model": JUDGE_MODEL["model_id"],
        "messages": [
            {"role": "system", "content": "You are an objective judge."},
            {"role": "user", "content": prompt}
        ]
    }

    headers = {
        "Authorization": f"Bearer {JUDGE_MODEL['token']}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(
            f"{JUDGE_MODEL['url']}/chat/completions",
            json=payload,
            headers=headers,
            timeout=30
        )
        resp.raise_for_status()
        score_text = resp.json()["choices"][0]["message"]["content"]
        return extract_score(score_text)
    except Exception as e:
        logger.warning(f"Judge model call failed: {e}, falling back to random score")
        return random.uniform(0.0, 1.0)

# --------------------------------
# Compute weights
# --------------------------------
def compute_weights(scores: Dict[str, list[float]]) -> Dict[str, float]:
    return {
        model: max(0.1, sum(vals) / len(vals))
        for model, vals in scores.items()
    }

# --------------------------------
# Judge loop
# --------------------------------
def run_loop():
    global LAST_RUN_TS, LAST_WEIGHTS

    while True:
        logger.info("Starting evaluation cycle")
        start_ts = time.time()

        samples = fetch_recent_requests()
        if not samples:
            logger.info("No samples available for evaluation")
            time.sleep(EVAL_INTERVAL_SECONDS)
            continue

        scores = defaultdict(list)
        for s in samples:
            score = judge_response(s["user_input"], s["output"])
            scores[s["model"]].append(score)

            # Store individual score in SQLite
            with get_conn() as conn:
                c = conn.cursor()
                c.execute(
                    "INSERT INTO scores (request_id, model, score, timestamp) VALUES (?, ?, ?, ?)",
                    (s["request_id"], s["model"], score, start_ts)
                )
                conn.commit()

        weights = compute_weights(scores)

        # Update weights table for gateway (judge is the only writer)
        with get_conn() as conn:
            c = conn.cursor()
            for model, weight in weights.items():
                c.execute(
                    "INSERT INTO model_weights (model, weight, last_updated) VALUES (?, ?, ?)"
                    "ON CONFLICT(model) DO UPDATE SET weight=excluded.weight, last_updated=excluded.last_updated",
                    (model, weight, start_ts)
                )
            conn.commit()

        LAST_RUN_TS = start_ts
        LAST_WEIGHTS = weights
        logger.info(f"Published new weights: {weights}")

        time.sleep(EVAL_INTERVAL_SECONDS)

# --------------------------------
# API endpoints (observability)
# --------------------------------
@app.get("/ping")
def ping():
    return {"ok": True}

@app.get("/status")
def status():
    return {
        "last_run_ts": LAST_RUN_TS,
        "eval_interval_seconds": EVAL_INTERVAL_SECONDS,
    }

@app.get("/weights")
def weights():
    return LAST_WEIGHTS or {}

# --------------------------------
# Server launcher
# --------------------------------
def run_server():
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.environ["CDSW_APP_PORT"]),
        log_level="warning",
    )

if __name__ == "__main__":
    threading.Thread(target=run_loop, daemon=True).start()
    threading.Thread(target=run_server, daemon=True).start()
    while True:
        time.sleep(60)
