# gpuest

Creates a local database of GPU energy consumption tracking.
Supports nvidia and amd vendors only, but may be extended to more.
There are three steps to energy estimation:

1. Record an idle energy profile (needed only if you need to subtract idle times from the next step)
2. Record a workflow profile that outputs the task magnitude (e.g., number of tokens)
3. Create an estimation of energy spent for new tasks

Results verify setup compatibility, and provide both mean estimation and confidence margins.

## :zap: Quickstart

Install this as a library per `pip install gpuest`.

Then, create an initial energy profile for the machine's idle state.
Energy profiles are stored in a file *energy_profiles.toml* that can
be shared between the exact same setups. This is done with the idiom
below, where the idle function needs to return its own sleep duration.

We will be using a helper function recording the idle profile, as
that can be tricky to do robustly from the get go; the helper
polls the GPU avery fixed number of seconds,
and gets the minimum spent energy across ten such intervals.
This avoids being influenced by energy spikes. The process is
repeated to get a sense of idle estimation standard deviation.
Make sure that the GPU is not utilized too much.

```python
# 1_profile_idle.py
import time
from gpuest import Energy

energy = Energy("energy_profiles.toml")
energy.record_idle(intervals=5.0, batches=10, repetitions_per_batch=10)
```

Create an energy profile for your GPU compute, such as an LLM.
Here we query an ollama model. The function passed to the profile should output
some notion of near-linearly increasing magnitude for a task using the GPU, like 
the number of tokens.

You can make multiple recordings, for example corresponding to different prompts,
and they aggregate (pass argument `reset=True` to overwrite previous values).

```python
# 2_profile.py
import requests  # pip install requests
from gpuest import Energy

model_name = "llama3.2:latest"
def ollama(prompt: str):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model_name, "prompt": prompt, "stream": False})
    if response.status_code != 200: return 0  # skip this measurement
    result = response.json()
    return result["eval_count"]

energy = Energy()
for prompt in [
    "Who am I?",
    "To be or not to be?"
]: energy.record(model_name, lambda: ollama(prompt), batches=5, repetitions_per_batch=5)
```

Now, to estimate the energy consumption for the gathered energy profile,
just run the following:

```python
# 3_estimate.py
import requests # pip install requests
from gpuest.energy import Energy

model_name = "llama3.2:latest"
def get_response(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model_name, "prompt": prompt, "stream": False})
    assert response.status_code==200
    return response.json()

result = get_response("What is CERTH? Explain in one sentence.")

energy = Energy()
magnitude = result["eval_count"]
profile_vendor, estimation_mean, estimation_margin = energy.estimate("llama3.2:latest", magnitude)

print("Response: "+result["response"])
print(f"Energy cost for {profile_vendor} estimated {estimation_mean-estimation_margin:.3f} to {estimation_mean+estimation_margin:.3f} Joule consumption")
```
