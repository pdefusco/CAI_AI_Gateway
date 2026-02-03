from fastapi import FastAPI, HTTPException, Request
import os
import requests
import uvicorn
import threading
import logging
import sys
import time

logger = logging.getLogger("ai_gateway")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"
)
handler.setFormatter(formatter)

logger.handlers = []        # avoid duplicate logs
logger.addHandler(handler)
logger.propagate = False

app = FastAPI()

# ------------------------------
# Configuration
# ------------------------------
MODELS = {
    "model-a": {
        "model_id": os.getenv("MODEL_A_ID"),
        "token": os.getenv("MODEL_A_TOKEN"),
        "url": os.getenv("MODEL_A_URL"),  # must end with /v1
    },
    "model-b": {
        "model_id": os.getenv("MODEL_B_ID"),
        "token": os.getenv("MODEL_B_TOKEN"),
        "url": os.getenv("MODEL_B_URL"),  # must end with /v1
    },
}

# ------------------------------
# Forwarding function (SYNC, requests)
# ------------------------------
def forward_to_cloudera(model_id: str, base_url: str, token: str, user_input: str):
    url = f"{base_url}/chat/completions"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": user_input}
        ]
    }

    logger.info(f"Calling model='{model_id}' url='{url}'")
    logger.info(f"Prompt: {user_input}")

    start_time = time.time()

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
    except requests.RequestException as e:
        logger.error(f"Request to Cloudera failed: {e}")
        raise HTTPException(status_code=502, detail=str(e))

    duration = round(time.time() - start_time, 2)
    logger.info(f"Cloudera response status={response.status_code} time={duration}s")

    if response.status_code != 200:
        logger.error(f"Cloudera error body: {response.text}")
        raise HTTPException(
            status_code=response.status_code,
            detail=response.text
        )

    data = response.json()
    output = data["choices"][0]["message"]["content"]

    # Avoid massive logs (truncate)
    logger.info(f"Model output (first 500 chars): {output[:500]}")

    return {"output": output}


# ------------------------------
# Endpoints
# ------------------------------
@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/ping")
def ping():
    return {"ok": True}

@app.post("/inference")
async def inference(request: Request):
    payload = await request.json()

    logger.info(f"Incoming request: {json.dumps(payload, indent=2)}")

    model_name = payload.get("model_name")
    if not model_name:
        raise HTTPException(status_code=400, detail="Missing 'model_name' field")

    model_info = MODELS.get(model_name)
    if not model_info:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    user_input = payload.get("inputs")
    if not user_input:
        raise HTTPException(status_code=400, detail="Missing 'inputs' field")

    logger.info(f"Routing request to model '{model_name}'")

    return forward_to_cloudera(
        model_id=model_info["model_id"],
        base_url=model_info["url"],
        token=model_info["token"],
        user_input=user_input
    )

# ------------------------------
# Run server
# ------------------------------
def run_server():
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.environ["CDSW_APP_PORT"]),
        log_level="warning"
    )

threading.Thread(target=run_server).start()
