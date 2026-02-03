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
# Model endpoints
# -------------------------------------------------------------------

@app.post("/model-a")
async def model_a(request: Request):
    payload = await request.json()
    return await forward_to_cloudera(MODEL_A_URL, payload, MODEL_A_TOKEN)

@app.post("/model-b")
async def model_b(request: Request):
    payload = await request.json()
    return await forward_to_cloudera(MODEL_B_URL, payload, MODEL_B_TOKEN)

# -------------------------------------------------------------------
# Uvicorn entry point
# -------------------------------------------------------------------

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ['CDSW_APP_PORT']), log_level="warning", reload=False)

server_thread = threading.Thread(target=run_server)
server_thread.start()
