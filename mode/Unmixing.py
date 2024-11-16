from core import ModeAdapter
from .run import RunMode
from .params import ParamsMode


class Unmixing:
    def __init__(self, data=None):
        self.r = RunMode()
        self.p = ParamsMode()

    def __call__(self):
        cp = ModeAdapter()
        if cp.cfg.mode == "run":
            self.r.run()
        elif cp.cfg.mode == "params":
            self.p.run()
        else:
            print("Check the 'mode' field in main_config.yaml, and maybe there's something wrong!")
