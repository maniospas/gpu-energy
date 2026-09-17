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
print(f"Total real energy cost: {energy_spent:.3f} (THIS INCLUDES THE IDLE COST THAT THE ABOVE ESTIMATION EXCLUDES)")
