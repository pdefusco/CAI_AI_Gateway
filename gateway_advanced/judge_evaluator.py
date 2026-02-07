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

def log_text_block(title: str, text: str, max_chars: int = 500):
    clipped = text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]"
    logger.info(f"{title}:\n{clipped}")

# --------------------------------
# Config
# --------------------------------
EVAL_INTERVAL_SECONDS = 30

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
DB_PATH = "/tmp/requests.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=60, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=DELETE;")
    return conn

# Initialize tables
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
    for model in ["model-a", "model-b"]:
        c.execute(
            "INSERT OR IGNORE INTO model_weights (model, weight) VALUES (?, ?)",
            (model, 1.0)
        )

    c.execute("""
        CREATE TABLE IF NOT EXISTS model_weights_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT,
            weight REAL,
            last_updated REAL
        )
        """)

    conn.commit()

# --------------------------------
# Fetch requests
# --------------------------------
def fetch_recent_requests():
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT request_id, model_chosen, user_input, model_output
            FROM requests
            WHERE model_output IS NOT NULL AND judged_at IS NULL
        """)
        rows = c.fetchall()

    return [
        {
            "request_id": r[0],
            "model": r[1],
            "user_input": r[2],
            "output": r[3],
        }
        for r in rows
    ]

# --------------------------------
# Judge scoring
# --------------------------------
def extract_score(score_text: str) -> float:
    match = re.findall(r"([0-1](?:\.\d+)?)", score_text)
    if match:
        score = float(match[-1])
        return min(max(score, 0.0), 1.0)
    logger.warning("No numeric score found in judge response; using random fallback")
    return random.uniform(0.0, 1.0)

def judge_response(user_input: str, model_output: str):
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
            {"role": "user", "content": prompt},
        ],
    }

    headers = {
        "Authorization": f"Bearer {JUDGE_MODEL['token']}",
        "Content-Type": "application/json",
    }

    try:
        start = time.time()
        resp = requests.post(
            f"{JUDGE_MODEL['url']}/chat/completions",
            json=payload,
            headers=headers,
            timeout=180,
        )
        latency = time.time() - start

        resp.raise_for_status()
        score_text = resp.json()["choices"][0]["message"]["content"]
        score = extract_score(score_text)

        return score, score_text, latency

    except Exception as e:
        logger.warning(f"Judge model call failed: {e}")
        fallback = random.uniform(0.0, 1.0)
        return fallback, "ERROR_FALLBACK", 0.0

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
        logger.info("=" * 80)
        logger.info("Starting evaluation cycle")
        start_ts = time.time()

        samples = fetch_recent_requests()
        if not samples:
            logger.info("No samples available for evaluation")
            time.sleep(EVAL_INTERVAL_SECONDS)
            continue

        logger.info(f"Evaluating {len(samples)} samples")

        scores = defaultdict(list)

        for s in samples:
            logger.info("-" * 60)
            logger.info(f"Judging request_id={s['request_id']} model={s['model']}")

            log_text_block("Question", s["user_input"])
            log_text_block("Answer", s["output"])

            score, raw_judgment, latency = judge_response(
                s["user_input"], s["output"]
            )

            log_text_block("Judge raw output", raw_judgment, max_chars=200)
            logger.info(
                f"Parsed score={round(score, 3)} (judge latency={round(latency, 2)}s)"
            )

            scores[s["model"]].append(score)

            with get_conn() as conn:
                c = conn.cursor()
                c.execute(
                    "INSERT INTO scores (request_id, model, score, timestamp) VALUES (?, ?, ?, ?)",
                    (s["request_id"], s["model"], score, start_ts),
                )
                conn.commit()

        weights = compute_weights(scores)

        with get_conn() as conn:
            c = conn.cursor()
            for model, weight in weights.items():
                c.execute(
                    """
                    INSERT INTO model_weights (model, weight, last_updated)
                    VALUES (?, ?, ?)
                    ON CONFLICT(model)
                    DO UPDATE SET weight=excluded.weight,
                                  last_updated=excluded.last_updated
                    """,
                    (model, weight, start_ts),
                )

            conn.commit()

        with get_conn() as conn:
            c = conn.cursor()
            for model, weight in weights.items():
                c.execute(
                    "INSERT INTO model_weights_history (model, weight, last_updated) VALUES (?, ?, ?)",
                    (model, weight, start_ts)
                )

            conn.commit()


        LAST_RUN_TS = start_ts
        LAST_WEIGHTS = weights

        logger.info("-" * 60)
        logger.info(f"Published new weights: {weights}")

        time.sleep(EVAL_INTERVAL_SECONDS)

# --------------------------------
# API endpoints
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
