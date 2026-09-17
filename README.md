# GPU energy estimator

Creates a local database of GPU energy consumption tracking.
Supports nvidia and amd vendors only, but may be extended to more.
There are three steps to energy estimation:

1. Record an idle energy profile (needed only if you need to subtract idle times from the next step)
2. Record a workflow profile that outputs the task magnitude (e.g., number of tokens)
3. Create an estimation of energy spent for new tasks

Results verify setup compatibility, and provide both mean estimation and confidence margins.

## :zap: Quickstart

Clone this repo and install dependencies per `pip install nvidia-ml-py tomli-w`.

Then, create an initial energy profile for the machine's idle state.
Energy profiles are stored in a file *energy_profiles.toml* that can
be shared between the exact same setups. This is done with the idiom
below, where the idle function needs to return its own sleep duration.

```python
# 1_profile_idle.py
import time
from energy.energy import Energy

def idle(duration: float):
    time.sleep(duration)
    return duration

energy = Energy()
energy.record("idle", lambda: idle(duration=10.0), repetitions=20)
```

Create similarly an energy profile for your GPU compute, such as an LLM.
Here we query an ollama model. The function passed to the profile should output
some notion of near-linearly increasing magnitude for a task using the GPU, like 
the number of tokens.

You can make multiple recordings, and they aggregate (use `energy.clear(profile_name)` 
to clear a profile's data):

```python
# 2_profile.py
import requests # pip install requests
from energy.energy import Energy

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
]: energy.record(model_name, lambda: ollama(prompt), repetitions=5, time_profile="idle")
```

Now, to estimate the energy consumption for the gathered energy profile,
just run the following:

```python
# 3_estimate.py
import requests # pip install requests
from energy.energy import Energy

model_name = "llama3.2:latest"
def get_response(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model_name, "prompt": prompt, "stream": False})
    assert response.status_code==200
    return response.json()

get_response("Hi.") # hot start

energy = Energy()
prev_accumulated_energy = energy.vendor.get_gpu_accumulation()
result = get_response("What is CERTH? Explain in one sentence.")
energy_spent = energy.vendor.get_gpu_accumulation()-prev_accumulated_energy

magnitude = result["prompt_eval_count"]+result["eval_count"]
profile_vendor, estimation_mean, estimation_margin = energy.estimate("llama3.2:latest", magnitude)

print("Response: "+result["response"])
print(f"Energy cost for {profile_vendor} estimated {estimation_mean-estimation_margin:.3f} to {estimation_mean+estimation_margin:.3f} Joule consumption")
print(f"Total real energy cost: {energy_spent:.3f} (THIS INCLUDES THE IDLE AND OTHER WORKLOAD COST THAT THE ABOVE ESTIMATION EXCLUDES)")
```
