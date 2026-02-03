from fastapi import FastAPI, HTTPException, Request
import os
import uvicorn
import threading
from langchain_openai import ChatOpenAI
import json

# ------------------------------
# FastAPI app
# ------------------------------

app = FastAPI()

# ------------------------------
# Configuration
# ------------------------------

MODEL_A_TOKEN = os.getenv("MODEL_A_TOKEN")
MODEL_B_TOKEN = os.getenv("MODEL_B_TOKEN")

MODEL_A_ID = os.getenv("MODEL_A_ID")  # nvidia/llama-3.3-nemotron-super-49b-v1
MODEL_B_ID = os.getenv("MODEL_B_ID")  # defog/llama-3-sqlcoder-8b

MODEL_A_URL = os.getenv("MODEL_A_URL")
MODEL_B_URL = os.getenv("MODEL_B_URL")

# ------------------------------
# MODELS dictionary
# ------------------------------

MODELS = {
    "model-a": {
        "model_id": MODEL_A_ID,
        "token": MODEL_A_TOKEN,
        "url": MODEL_A_URL,
    },
    "model-b": {
        "model_id": MODEL_B_ID,
        "token": MODEL_B_TOKEN,
        "url": MODEL_B_URL,
    },
}

# ------------------------------
# Forwarding function (sync)
# ------------------------------

def forward_to_cloudera(model_id: str, base_url: str, token: str, payload: dict):
    user_input = payload.get("inputs")
    if not user_input:
        raise HTTPException(status_code=400, detail="Missing 'inputs' field")

    # Instantiate the LLM
    llm = ChatOpenAI(
        model=model_id,
        api_key=token,
        base_url=base_url,
        temperature=0.0,
    )

    try:
        # Call synchronously with just text
        output = llm(user_input)  # returns string
        return {"output": output}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

# ------------------------------
# FastAPI endpoints
# ------------------------------

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/ping")
def ping():
    return {"ok": True}

@app.post("/inference")
def inference(request: Request):
    payload = json.loads(request.body().read())  # sync read

    model_name = payload.get("model_name")
    if not model_name:
        raise HTTPException(status_code=400, detail="Missing 'model_name' field")

    model_info = MODELS.get(model_name)
    if not model_info:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    payload.pop("model_name")

    response = forward_to_cloudera(
        model_id=model_info["model_id"],
        base_url=model_info["url"],
        token=model_info["token"],
        payload=payload
    )
    return response

# ------------------------------
# Run server
# ------------------------------

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ['CDSW_APP_PORT']), log_level="warning")

threading.Thread(target=run_server).start()
