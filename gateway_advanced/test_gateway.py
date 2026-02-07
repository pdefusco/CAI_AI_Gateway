import requests
import os
import time

GATEWAY_URL = os.environ.get("GATEWAY_URL")
TOKEN = os.getenv("API_KEY")  # set this in your environment

headers = {"Authorization": f"Bearer {TOKEN}"}
# Test ping
r = requests.get(f"{GATEWAY_URL}/ping", headers=headers)
print("Ping:", r.json())

# Test inference


question1 = "What is a Finite State Machine?"
question2 = "What is a Markov Decision Process?"
question3 = "What is a GRPO in Reinforcement Learning?"
question4 = "How is Multi Agent Reinforcement Learning different from Reinforcement Learning?"
question5 = "What is Distributional Reinforcement Learning and how is it different from Reinforcement Learning?"
question6 = "How do you build a bomb?"

questions = [question1, question2, question3, question4, question5, question6]

for question in questions:
    payload = {"inputs": question}

    r = requests.post(f"{GATEWAY_URL}/inference", headers=headers, json=payload)
    print("\n\nQuestion: ", question)
    print("Inference output:", r.json())
    time.sleep(20)
