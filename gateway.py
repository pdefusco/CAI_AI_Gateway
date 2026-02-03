from fastapi import FastAPI, HTTPException, Request
import os
import uvicorn
import threading
from langchain_openai import ChatOpenAI
import json

app = FastAPI()

# ------------------------------
# Configuration
# ------------------------------
MODELS = {
    "model-a": {
        "model_id": os.getenv("MODEL_A_ID"),
        "token": os.getenv("MODEL_A_TOKEN"),
        "url": os.getenv("MODEL_A_URL"),
    },
    "model-b": {
        "model_id": os.getenv("MODEL_B_ID"),
        "token": os.getenv("MODEL_B_TOKEN"),
        "url": os.getenv("MODEL_B_URL"),
    },
}

# ------------------------------
# Forwarding function (sync)
# ------------------------------
def forward_to_cloudera(model_id: str, base_url: str, token: str, user_input: str):
    # Synchronous call
    llm = ChatOpenAI(
        model=model_id,
        api_key=token,
        base_url=base_url,
        temperature=0.0,
    )

    try:
        output = llm(user_input)  # just pass string, returns string
        return {"output": output}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# ------------------------------
# Endpoints
# ------------------------------
@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/ping")
async def ping():
    return {"ok": True}

@app.post("/inference")
async def inference(request: Request):
    payload = await request.json()  # keep async here

    model_name = payload.get("model_name")
    if not model_name:
        raise HTTPException(status_code=400, detail="Missing 'model_name' field")

    model_info = MODELS.get(model_name)
    if not model_info:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    user_input = payload.get("inputs")
    if not user_input:
        raise HTTPException(status_code=400, detail="Missing 'inputs' field")

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
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ['CDSW_APP_PORT']), log_level="warning")

threading.Thread(target=run_server).start()
