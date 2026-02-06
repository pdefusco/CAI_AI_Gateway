import requests
import os

port = os.environ.get("CDSW_APP_PORT", "8080")

# Test ping
r = requests.get(f"http://127.0.0.1:{port}/ping")
print("Ping:", r.json())

# Test inference
r = requests.post(f"http://127.0.0.1:{port}/inference", json={"inputs": "What is a finite state machine?"})
print("Inference output:", r.json())
