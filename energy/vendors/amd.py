from energy.energy import GPUVendor

class AMDVendor(GPUVendor):
    def __init__(self):
        super().__init__()
        self.amdsmi = None

    def _lazy_init(self):
        if self.amdsmi is not None: return
        import amdsmi
        amdsmi.amdsmi_init()
        self.gpus = amdsmi.amdsmi_get_processor_handles()
        self.amdsmi = amdsmi

    def get_gpu_accumulation(self) -> float:
        self._lazy_init()
        total_uj = 0.0
        for gpu in self.gpus:
            energy = self.amdsmi.amdsmi_get_energy_count(gpu)
            total_uj +=  energy["energy_accumulator"] * energy["counter_resolution"]
        return total_uj / 1_000_000.0