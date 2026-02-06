from fastapi import FastAPI
import json
import os

app = FastAPI()
WEIGHT_ARTIFACT_PATH = "/tmp/model_weights.json"


@app.get("/weights")
def weights():
    if not os.path.exists(WEIGHT_ARTIFACT_PATH):
        return {"status": "no artifact"}
    with open(WEIGHT_ARTIFACT_PATH) as f:
        return json.load(f)


@app.get("/health")
def health():
    return {"ok": True}
