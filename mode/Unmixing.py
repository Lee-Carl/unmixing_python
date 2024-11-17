from core import ModeAdapter
from .run import RunMode
from .params import ParamsMode
from custom_types import MainCfg, ModeEnum


class Unmixing:
    def __init__(self, dic: dict):
        self.cfg = MainCfg(**dic)

    def getMode(self):
        if self.cfg.mode == ModeEnum.Run:
            return RunMode(self.cfg)
        if self.cfg.mode == ModeEnum.Param:
            return ParamsMode(self.cfg)
        print("Check the 'mode' field in main_config.yaml, and maybe there's something wrong!")

    def __call__(self):
        self.getMode().run()
