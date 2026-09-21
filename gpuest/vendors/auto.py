from gpuest.energy import GPUVendor
from contextlib import redirect_stdout, redirect_stderr
import io

class AutoVendor(GPUVendor):
    def __init__(self, vendors:list[GPUVendor]|None=None, verbose=True):
        super().__init__()
        if vendors is None:
            from gpuest.vendors.amd import AMDVendor
            from gpuest.vendors.nvidia import NvidiaVendor
            vendors = [AMDVendor(), NvidiaVendor()]
        self.vendors: list[GPUVendor] = vendors
        self.vendor: GPUVendor|None = None
        self.verbose = verbose

    def _lazy_init(self):
        if self.vendor is not None: return
        print("AutoVendor is going through known vendors...")
        for vendor in self.vendors:
            if  self.verbose: print(f"- {vendor.who()} ", end="", flush=True)
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    vendor._lazy_init()
            except:
                if self.verbose: print("not found")
                continue
            if self.verbose: print("success!")
            self.vendor = vendor
            break
        if self.vendor is None: raise Exception("No vendor found")

    def get_gpu_accumulation(self):
        self._lazy_init()
        assert self.vendor is not None
        return self.vendor.get_gpu_accumulation()

    def who(self):
        self._lazy_init()
        assert self.vendor is not None
        return self.vendor.who()