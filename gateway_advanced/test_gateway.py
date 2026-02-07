import requests
import os

GATEWAY_URL = os.environ.get("GATEWAY_URL")
TOKEN = os.getenv("API_KEY")  # set this in your environment

headers = {"Authorization": f"Bearer {TOKEN}"}
# Test ping
r = requests.get(f"{GATEWAY_URL}/ping", headers=headers)
print("Ping:", r.json())

# Test inference

question = "What is a finite state machine?"
payload = {"inputs": question}

r = requests.post(f"{GATEWAY_URL}/inference", headers=headers, json=payload)
print("Inference output:", r.json())
