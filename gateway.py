from fastapi import FastAPI, HTTPException, Request
import httpx
import os
import uvicorn
import threading

app = FastAPI(title="Trylon AI Gateway")

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

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                json=payload,
                headers=HEADERS
            )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Cloudera request failed: {exc}"
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=response.text
        )

    return response.json()

# -------------------------------------------------------------------
# Model endpoint
# -------------------------------------------------------------------

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

    model_name = payload.get("model_name")
    if not model_name:
        raise HTTPException(status_code=400, detail="Missing 'model_name' field")

    model_info = MODELS.get(model_name)
    if not model_info:
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
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ['CDSW_APP_PORT']), log_level="warning", reload=False)

server_thread = threading.Thread(target=run_server)
server_thread.start()
