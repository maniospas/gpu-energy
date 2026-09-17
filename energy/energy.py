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
            from energy.vendors.auto import AutoVendor
            vendor = AutoVendor()
        self.vendor: GPUVendor = vendor

    def _load_profiles(self):
        if not self.profile_path.exists(): return
        with self.profile_path.open("rb") as f: data = tomllib.load(f)
        self.profiles = data.get("profiles", {})

    def _save_profiles(self):
        data = {"profiles": self.profiles}
        with self.profile_path.open("wb") as f:tomli_w.dump(data, f)
    def reset(self, name: str): del self.profiles[name]
    def record(
            self,
            name: str,
            function: Callable[[], float],
            repetitions: int = 1,
            time_profile: str | None = None,
    ) -> dict:
        assert repetitions>=1
        profile = self.profiles.get(name, {
            "mean": 0.0,
            "std": 0.0,
            "count": 0,
            "vendor": ""
        })
        count = int(profile["count"])
        mean = float(profile["mean"])
        std = float(profile["std"])
        m2 = std ** 2 * (count - 1) if count > 1 else 0.0
        if time_profile is not None:
            base_profile = self.profiles[str(time_profile)]
            assert base_profile["vendor"] == self.vendor.who(), f"Cannot combine time profile obtained for vendor {base_profile['vendor']} in vendor {self.vendor.who()}"
        assert profile["vendor"]==self.vendor.who() or not profile["vendor"], f"Cannot combine energy estimates between vendors {profile['vendor']} and {self.vendor.who()}"

        for _ in range(repetitions):
            energy_before = self.vendor.get_gpu_accumulation()
            time_before = time.perf_counter()
            magnitude = function()
            if magnitude==0: continue
            elapsed = time.perf_counter() - time_before
            energy = self.vendor.get_gpu_accumulation()-energy_before
            magnitude = float_positive(magnitude)
            if time_profile is not None: energy -= float(base_profile["mean"]) * elapsed
            sample = energy / magnitude
            count += 1
            delta = float(sample - mean)
            mean += delta / count
            delta2 = sample - mean
            m2 += delta * delta2
        std = ((m2 / (count - 1))**0.5 if count > 1 else 0.0)
        self.profiles[name] = {
            "mean": mean,
            "std": std,
            "count": count,
            "vendor": self.vendor.who()
        }
        self._save_profiles()
        return self.profiles[name]

    def estimate(
            self,
            name: str,
            magnitude: float,
            confidence: float = 0.95,
    ) -> tuple[str, float, float]:
        magnitude = float_positive(magnitude)
        profile = self.profiles[str(name)]
        count = int(profile["count"])

        estimated_energy = magnitude * float(profile["mean"])
        if count < 2: return profile["vendor"], estimated_energy, 0.0
        standard_error = float(profile["std"]) / count**0.5

        z = NormalDist().inv_cdf(0.5 + 0.5*float_unit(confidence))
        margin = z * standard_error * magnitude
        return profile["vendor"], estimated_energy, margin
