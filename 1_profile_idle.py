from gpuest import Energy

energy = Energy("energy_profiles.toml")
energy.record_idle(intervals=5.0, batches=10, repetitions_per_batch=10)
