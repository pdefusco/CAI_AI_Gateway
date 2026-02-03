from fastapi import FastAPI, HTTPException, Request
import httpx
import os
import uvicorn
import threading
import json
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

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

MODEL_A_ID = os.getenv("MODEL_A_ID") #nvidia/llama-3.3-nemotron-super-49b-v1
MODEL_B_ID = os.getenv("MODEL_B_ID") #defog/llama-3-sqlcoder-8b

MODEL_A_URL = os.getenv("MODEL_A_URL")
MODEL_B_URL = os.getenv("MODEL_B_URL")

# -------------------------------------------------------------------
# Shared forwarding logic (gateway core)
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Shared forwarding logic using LangChain
# -------------------------------------------------------------------

async def forward_to_cloudera(model_id: str, base_url: str, token: str, payload: dict):
    """
    Forward request to the chosen Cloudera AI model using LangChain's ChatOpenAI wrapper.
    Uses per-model base_url.
    """
    logger.info(f"Forwarding request to Cloudera model '{model_id}' at '{base_url}'")

    user_input = payload.get("inputs")
    if not user_input:
        logger.warning("Missing 'inputs' in payload")
        raise HTTPException(status_code=400, detail="Missing 'inputs' field")

    # Instantiate the LLM with the correct base_url for this model
    llm = ChatOpenAI(
        model=model_id,
        api_key=token,
        base_url=base_url,
        temperature=0.0,
    )

    try:
        response = llm([HumanMessage(content=user_input)])
        output = response.content
        logger.info(f"Received response from model '{model_id}'")
        return {"output": output}
    except Exception as e:
        logger.error(f"Error calling model '{model_id}': {e}")
        raise HTTPException(status_code=502, detail=str(e))

# -------------------------------------------------------------------
# MODELS dictionary (include per-model URL)
# -------------------------------------------------------------------

MODELS = {
    "model-a": {
        "model_id": os.getenv("MODEL_A_ID"),
        "token": os.getenv("MODEL_A_TOKEN"),
        "url": os.getenv("MODEL_A_URL"),  # this will be passed as base_url
    },
    "model-b": {
        "model_id": os.getenv("MODEL_B_ID"),
        "token": os.getenv("MODEL_B_TOKEN"),
        "url": os.getenv("MODEL_B_URL"),  # this will be passed as base_url
    },
}

# -------------------------------------------------------------------
# /inference endpoint
# -------------------------------------------------------------------

@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/ping")
async def ping():
    logger.info("Ping endpoint hit")
    return {"ok": True}


@app.post("/inference")
async def inference(request: Request):
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

    # Remove 'model_name' before sending to the model
    payload.pop("model_name")

    return await forward_to_cloudera(
        model_id=model_info["model_id"],
        base_url=model_info["url"],
        token=model_info["token"],
        payload=payload
    )
# -------------------------------------------------------------------
# Uvicorn entry point
# -------------------------------------------------------------------

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ['CDSW_APP_PORT']), log_level="warning", reload=False)

server_thread = threading.Thread(target=run_server)
server_thread.start()
