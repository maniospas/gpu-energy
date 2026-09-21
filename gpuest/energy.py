import time
import tomllib
import tomli_w
from typing import Callable
from pathlib import Path
from statistics import NormalDist

def float_unit(x):
    x = float(x)
    assert 0.0<=x<=1.0
    return x

def float_positive(x):
    x = float(x)
    assert x>0.0
    return x

class GPUVendor:
    def __init__(self): pass
    def _lazy_init(self)->None: pass
    def get_gpu_accumulation(self)->float:
        """Return cumulative energy consumption across all GPUs in joules."""
        raise NotImplementedError
    def who(self)->str: return self.__class__.__name__

class Energy:
    def __init__(self, profile_path="energy_profiles.toml", vendor:GPUVendor|None=None):
        self.profile_path = Path(profile_path)
        self.profiles = {}
        self._load_profiles()
        if vendor is None:
            from gpuest.vendors.auto import AutoVendor
            vendor = AutoVendor()
        self.vendor: GPUVendor = vendor

    def _load_profiles(self):
        if not self.profile_path.exists(): return
        with self.profile_path.open("rb") as f: data = tomllib.load(f)
        self.profiles = data.get("profiles", {})

    def _save_profiles(self):
        data = {"profiles": self.profiles}
        with self.profile_path.open("wb") as f:tomli_w.dump(data, f)

    def reset(self, name: str):
        del self.profiles[name]

    def record_idle(self, intervals=5.0, batches=10, repetitions_per_batch=10, reset=False):
        def idle(duration: float):
            time.sleep(duration)
            return duration
        def min_first(pairs):
            return min(pairs, key=lambda pair: pair[0])
        for repetition in range(batches):  # 10 times get the minimum energy consumption across 10 runs (avoid spikes)
            self.raw_record(
                "idle",
                lambda: idle(duration=intervals),
                repetitions=repetitions_per_batch,
                reset=reset and repetition == 0,
                batch_aggregate=min_first)

    def record(
            self,
            name: str,
            function: Callable[[], float],
            batches=10,
            repetitions_per_batch: int = 10,
            reset: bool = False,
    ) -> dict:
        assert name!="idle"
        def min_first(pairs):
            return min(pairs, key=lambda pair: pair[0])
        for repetition in range(batches):  # 10 times get the minimum energy consumption across 10 runs (avoid spikes)
            self.raw_record(
                name,
                function,
                repetitions=repetitions_per_batch,
                reset=reset and repetition == 0,
                batch_aggregate=min_first,
                time_profile="idle")

    def raw_record(
            self,
            name: str,
            function: Callable[[], float],
            repetitions: int = 1,
            time_profile: str | None = None,
            reset: bool = False,
            batch_aggregate: Callable[[list[tuple[float, float]]], tuple[float, float]] | None = None,
    ) -> dict:
        assert repetitions >= 1
        vendor = self.vendor.who()
        recordings = self.profiles.setdefault(name, {})
        profile = recordings.get(None if reset else vendor, {
            "mean": 0.0,
            "std": 0.0,
            "time_std": 0.0,
            "count": 0,
        })
        count = int(profile["count"])
        mean = float(profile["mean"])
        std = float(profile["std"])
        base_variance = float(profile.get("time_std", 0.0)) ** 2
        m2 = std ** 2 * (count - 1) if count > 1 else 0.0
        if time_profile is not None:
            base_profile = self.profiles[str(time_profile)][vendor]

        samples = []
        for _ in range(repetitions):
            energy_before = self.vendor.get_gpu_accumulation()
            time_before = time.perf_counter()
            magnitude = function()
            if magnitude == 0: continue
            elapsed = time.perf_counter() - time_before
            energy = self.vendor.get_gpu_accumulation() - energy_before
            magnitude = float_positive(magnitude)
            sample_std = 0.0
            if time_profile is not None:
                energy -= float(base_profile["mean"]) * elapsed
                sample_std = float(base_profile["std"]) * elapsed / magnitude
            sample = energy / magnitude
            if batch_aggregate is not None:
                samples.append((sample, sample_std))
                continue
            count += 1
            delta = float(sample - mean)
            mean += delta / count
            delta2 = sample - mean
            m2 += delta * delta2
            base_variance += (sample_std ** 2 - base_variance) / count

        if batch_aggregate is not None:
            assert samples, "all sample recordings failed"
            sample, sample_std = batch_aggregate(samples)
            sample = float(sample)
            sample_std = float(sample_std)
            count += 1
            delta = float(sample - mean)
            mean += delta / count
            delta2 = sample - mean
            m2 += delta * delta2
            base_variance += (sample_std ** 2 - base_variance) / count

        std = ((m2 / (count - 1)) ** 0.5 if count > 1 else 0.0)
        recordings[vendor] = {
            "mean": mean,
            "std": std,
            "time_std": base_variance ** 0.5,
            "count": count,
        }
        self._save_profiles()
        return recordings[vendor]

    def estimate(
            self,
            name: str,
            magnitude: float,
            confidence: float = 0.999,
            vendor: str | None = None,
    ) -> tuple[str, float, float]:
        magnitude = float_positive(magnitude)
        if vendor is None: vendor = self.vendor.who()
        profile = self.profiles[str(name)][vendor]
        count = int(profile["count"])

        estimated_energy = magnitude * float(profile["mean"])
        if count < 2: return vendor, estimated_energy, 0.0

        std = float(profile["std"])
        time_std = float(profile.get("time_std", 0.0))
        standard_error = (std ** 2 + time_std ** 2) ** 0.5 / count ** 0.5

        z = NormalDist().inv_cdf(0.5 + 0.5 * float_unit(confidence))
        margin = z * standard_error * magnitude
        return vendor, estimated_energy, margin