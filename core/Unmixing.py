from .processor import CoreProcessor
from .mode import ParamsMode, RunMode


class Unmixing:
    def __init__(self):
        self.r = RunMode()
        self.p = ParamsMode()

    def __call__(self):
        cp = CoreProcessor()
        if cp.cfg.mode == "run":
            self.r.run()
        elif cp.cfg.mode == "params":
            self.p.run()
        else:
            print("Check the 'mode' field in main_config.yaml, and maybe there's something wrong!")
