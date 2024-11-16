from typing import List


class MainConfig_Init:
    def __init__(self, **kwargs):
        self.show_initdata: str = kwargs.get('show_initdata', False)
        self.custom_init_data: str = kwargs.get('custom_init_data', None)
        self.custom_init_method: str = kwargs.get('custom_init_method', None)
        self.snr: float = kwargs.get('snr', 0)
        self.normalization: bool = kwargs.get('normalization', False)
        self.A: str = kwargs.get('A', None)
        self.E: str = kwargs.get('E', None)
        self.D: str = kwargs.get('D', None)


class MainConfig_Output:
    def __init__(self, **kwargs):
        self.draw: bool = kwargs.get('draw', False)
        self.normalization: bool = kwargs.get('normalization', False)
        self.sort: bool = kwargs.get('sort', False)
        self.metrics: str = kwargs.get('metrics', None)


class MainConfig_Params:
    def __init__(self, **kwargs):
        self.obj: str = kwargs.get('obj', None)
        self.around: List[float] = eval(kwargs[kwargs['around']]) if kwargs['around'] else None


class MainCfg:
    def __init__(self, **kwargs):
        self.dataset: str = kwargs.get('dataset', None)
        self.init: MainConfig_Init = MainConfig_Init(**kwargs['init']) if kwargs['init'] else None
        self.mode: str = kwargs.get('mode', None)
        self.params: MainConfig_Params = MainConfig_Params(**kwargs['params']) if kwargs['params'] else None
        self.output: MainConfig_Output = MainConfig_Output(**kwargs['output']) if kwargs['output'] else None
        self.seed: int = kwargs.get('seed', 0)
