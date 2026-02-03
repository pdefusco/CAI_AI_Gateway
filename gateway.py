from fastapi import FastAPI, HTTPException, Request
import httpx
import os
import uvicorn
import threading
import json

# -------------------------------------------------------------------
# Configure logging
# -------------------------------------------------------------------

import logging
import sys

logger = logging.getLogger("trylon_gateway")
logger.setLevel(logging.INFO)

# Create handler to stdout
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False  # prevent duplicate logs


# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------

app = FastAPI()

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

MODEL_A_TOKEN = os.getenv("MODEL_A_TOKEN")
MODEL_B_TOKEN = os.getenv("MODEL_B_TOKEN")

MODEL_A_URL = os.getenv("MODEL_A_URL")
MODEL_B_URL = os.getenv("MODEL_B_URL")

# -------------------------------------------------------------------
# Shared forwarding logic (gateway core)
# -------------------------------------------------------------------

async def forward_to_cloudera(url: str, payload: dict, token: str):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    logger.info(f"Forwarding request to Cloudera model at {url}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                json=payload,
                headers=headers
            )
    except httpx.RequestError as exc:
        logger.error(f"Cloudera request failed: {exc}")
        raise HTTPException(
            status_code=502,
            detail=f"Cloudera request failed: {exc}"
        )

    if response.status_code != 200:
        logger.error(f"Cloudera returned error {response.status_code}: {response.text}")
        raise HTTPException(
            status_code=response.status_code,
            detail=response.text
        )

    logger.info(f"Cloudera response successful (status {response.status_code})")
    return response.json()

# -------------------------------------------------------------------
# Model endpoint
# -------------------------------------------------------------------

MODELS = {
    "model-a": {
        "url": MODEL_A_URL,
        "token": MODEL_A_TOKEN,
    },
    "model-b": {
        "url": MODEL_B_URL,
        "token": MODEL_B_TOKEN,
    },
}

@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/ping")
async def ping():
    logger.info("Ping endpoint hit")
    return {"ok": True}


@app.post("/inference")
async def inference(request: Request):
    """
    Example payload:
    {
        "model_name": "model-a",
        "inputs": "Explain finite state machines"
    }
    """
    payload = await request.json()

    logger.info(f"Incoming request:\n{json.dumps(payload, indent=2)}")

    print("Incoming request:", json.dumps(payload, indent=2))

    model_name = payload.get("model_name")
    if not model_name:
        logger.warning("Missing 'model_name' in request")
        raise HTTPException(status_code=400, detail="Missing 'model_name' field")

    model_info = MODELS.get(model_name)
    if not model_info:
        logger.warning(f"Unknown model requested: {model_name}")
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    # Remove 'model_name' before sending to Cloudera if not needed
    payload.pop("model_name")

    return await forward_to_cloudera(
        url=model_info["url"],
        payload=payload,
        token=model_info["token"]
    )
# -------------------------------------------------------------------
# Uvicorn entry point
# -------------------------------------------------------------------

def run_server():
    print("Running on port: ", os.environ["CDSW_APP_PORT"])
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ['CDSW_APP_PORT']), log_level="info", reload=False)

server_thread = threading.Thread(target=run_server)
server_thread.start()
