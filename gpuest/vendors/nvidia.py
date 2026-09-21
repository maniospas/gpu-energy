from gpuest.energy import GPUVendor

class NvidiaVendor(GPUVendor):
    def __init__(self):
        super().__init__()
        self.pynvml = None

    def _lazy_init(self):
        if self.pynvml is not None: return
        import pynvml
        pynvml.nvmlInit()
        self.gpus = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        self.pynvml = pynvml

    def get_gpu_accumulation(self):
        self._lazy_init()
        return sum(self.pynvml.nvmlDeviceGetTotalEnergyConsumption(g) for g in self.gpus) / 1000.0

    def who(self):
        self._lazy_init()
        names = [self.pynvml.nvmlDeviceGetName(g) for g in self.gpus]
        return f"nvidia ({', '.join(names)})"