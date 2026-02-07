from fastapi import FastAPI, HTTPException, Request
import os
import time
import random
import threading
import logging
import requests
import sqlite3
import uuid
from typing import Dict
import uvicorn
import json

# --------------------------------
# Logging
# --------------------------------
logger = logging.getLogger("ai_gateway")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.handlers = [handler]
logger.propagate = False

app = FastAPI()

# --------------------------------
# Guardrail
# --------------------------------
import re

FORBIDDEN_PATTERNS = [
    r"\bhow to build a bomb\b",
    r"\bcredit card numbers?\b",
    r"\bterrorist attack\b",
    r"\billegal drugs?\b",
]

def violates_policy(text: str) -> str | None:
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return pattern
    return None


# --------------------------------
# Model configuration (static)
# --------------------------------
MODELS = {
    "model-a": {
        "model_id": os.getenv("MODEL_A_ID"),
        "token": os.getenv("MODEL_A_TOKEN"),
        "url": os.getenv("MODEL_A_URL"),  # ends with /v1
        "cost": 1.0,
    },
    "model-b": {
        "model_id": os.getenv("MODEL_B_ID"),
        "token": os.getenv("MODEL_B_TOKEN"),
        "url": os.getenv("MODEL_B_URL"),
        "cost": 0.5,
    },
}

# --------------------------------
# Routing weights (dynamic)
# --------------------------------
MODEL_WEIGHTS: Dict[str, float] = {k: 1.0 for k in MODELS}
WEIGHT_REFRESH_SECONDS = 30

# --------------------------------
# SQLite helper
# --------------------------------
DB_PATH = "/home/cdsw/shared/requests.db"

def get_conn():
    """Return a new SQLite connection with timeout for concurrency"""
    conn = sqlite3.connect(DB_PATH, timeout=60, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=DELETE;")
    return conn

# Initialize tables
with get_conn() as conn:
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS requests (
        request_id TEXT PRIMARY KEY,
        user_input TEXT NOT NULL,
        model_chosen TEXT,
        model_output TEXT,
        timestamp REAL DEFAULT (strftime('%s','now'))
    )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS model_weights (
        model TEXT PRIMARY KEY,
        weight REAL NOT NULL,
        last_updated REAL DEFAULT (strftime('%s','now'))
    )
    """)
    for model in MODELS:
        c.execute(
            "INSERT OR IGNORE INTO model_weights (model, weight) VALUES (?, ?)",
            (model, 1.0)
        )
    conn.commit()

# --------------------------------
# Pretty logging helpers
# --------------------------------
def log_text_block(title: str, text: str, max_chars: int = 500):
    clipped = text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]"
    logger.info(f"{title}:\n{clipped}")

# --------------------------------
# Weight loading
# --------------------------------
def load_weights():
    global MODEL_WEIGHTS
    try:
        with get_conn() as conn:
            c = conn.cursor()
            c.execute("SELECT model, weight FROM model_weights")
            rows = c.fetchall()
            MODEL_WEIGHTS = {model: float(weight) for model, weight in rows}
            logger.info(f"Loaded model weights from DB: {MODEL_WEIGHTS}")
    except Exception as e:
        logger.error(f"Failed to load weights from DB: {e}")

def weight_refresher():
    while True:
        load_weights()
        time.sleep(WEIGHT_REFRESH_SECONDS)

threading.Thread(target=weight_refresher, daemon=True).start()

# --------------------------------
# Utilities
# --------------------------------
def weighted_choice(weights: Dict[str, float]) -> str:
    total = sum(weights.values())
    r = random.uniform(0, total)
    upto = 0
    for model, w in weights.items():
        upto += w
        if upto >= r:
            return model
    return random.choice(list(weights.keys()))

def forward_to_model(model_name: str, user_input: str):
    m = MODELS[model_name]
    url = f"{m['url']}/chat/completions"

    payload = {
        "model": m["model_id"],
        "messages": [{"role": "user", "content": user_input}],
    }

    headers = {
        "Authorization": f"Bearer {m['token']}",
        "Content-Type": "application/json",
    }

    start = time.time()
    resp = requests.post(url, json=payload, headers=headers, timeout=180)
    latency = time.time() - start

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=resp.text)

    output = resp.json()["choices"][0]["message"]["content"]

    return output, latency

# --------------------------------
# Endpoints
# --------------------------------
@app.get("/ping")
def ping():
    return {"ok": True}

@app.post("/inference")
async def inference(request: Request):
    body = await request.json()
    user_input = body.get("inputs")
    if not user_input:
        raise HTTPException(status_code=400, detail="Missing inputs")

    violation = violates_policy(user_input)
    if violation:
        logger.warning(json.dumps({
            "event": "policy_block",
            "reason": "forbidden_topic",
            "matched_pattern": violation,
            "prompt_preview": user_input[:200],
        }))

        raise HTTPException(
            status_code=403,
            detail="This request violates usage policies and cannot be processed."
        )

    request_id = str(uuid.uuid4())
    model = weighted_choice(MODEL_WEIGHTS)

    logger.info("=" * 80)
    logger.info(f"REQUEST id={request_id} → routing to {model}")
    log_text_block("Question", user_input)

    # Store request
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO requests (request_id, user_input) VALUES (?, ?)",
            (request_id, user_input)
        )
        conn.commit()

    output, latency = forward_to_model(model, user_input)

    log_text_block(
        f"Response (model={model}, latency={round(latency, 2)}s)",
        output
    )

    logger.info(json.dumps({
        "event": "inference_complete",
        "request_id": request_id,
        "model": model,
        "latency": round(latency, 3),
        "cost": MODELS[model]["cost"],
        "prompt_chars": len(user_input),
        "output_chars": len(output),
    }))

    # Store response
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            "UPDATE requests SET model_chosen=?, model_output=? WHERE request_id=?",
            (model, output, request_id)
        )
        conn.commit()

    return {
        "request_id": request_id,
        "model": model,
        "output": output,
    }

# --------------------------------
# Server runner
# --------------------------------
def run_server():
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.environ["CDSW_APP_PORT"]),
        log_level="warning",
    )

if __name__ == "__main__":
    threading.Thread(target=run_server, daemon=True).start()
    while True:
        time.sleep(60)
