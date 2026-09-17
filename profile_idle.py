import time
from energy.energy import Energy

def idle(duration: float):
    time.sleep(duration)
    return duration

energy = Energy("energy_profiles.toml")
energy.record("idle", lambda: idle(duration=5.0), repetitions=10)
