from fastapi import FastAPI, HTTPException, Request
import httpx
import os

app = FastAPI(title="Trylon AI Gateway")

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

CLOUDERA_TOKEN = os.getenv("CLOUDERA_TOKEN", "REPLACE_ME")

MODEL_A_URL = "https://cloudera-ai.example.com/model-a/inference"
MODEL_B_URL = "https://cloudera-ai.example.com/model-b/inference"

HEADERS = {
    "Authorization": f"Bearer {CLOUDERA_TOKEN}",
    "Content-Type": "application/json",
}

# -------------------------------------------------------------------
# Shared forwarding logic (gateway core)
# -------------------------------------------------------------------

async def forward_to_cloudera(url: str, payload: dict):
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
    return await forward_to_cloudera(MODEL_A_URL, payload)

@app.post("/model-b")
async def model_b(request: Request):
    payload = await request.json()
    return await forward_to_cloudera(MODEL_B_URL, payload)

# -------------------------------------------------------------------
# Uvicorn entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "gateway:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
