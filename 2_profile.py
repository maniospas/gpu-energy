# 2_profile.py
import requests # pip install requests
from gpuest.energy import Energy

model_name = "llama3.2:latest"

def ollama(prompt: str):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={ "model": model_name, "prompt": prompt, "stream": False})
    if response.status_code!=200: return 0 # skip this measurement
    result = response.json()
    return result["prompt_eval_count"]+result["eval_count"]

energy = Energy()
for prompt in [
    "Who am I?",
    "To be or not to be?"
]: energy.record(model_name, lambda: ollama(prompt), batches=5, repetitions_per_batch=5)
