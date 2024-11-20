from typing import List
from .enums import InitA_Enum, InitE_Enum, DatasetsEnum, ModeEnum, MethodsEnum
from .type_alias import NoneableString


class MainConfig_Init:
    def __init__(self, **kwargs):
        self.show_initdata: bool = kwargs.get('show_initdata', False)
        self.custom_init_data: NoneableString = kwargs.get('custom_init_data', None)
        self.custom_init_method: NoneableString = kwargs.get('custom_init_method', None)
        self.snr: float = kwargs.get('snr', 0)
        self.normalization: bool = kwargs.get('normalization', False)
        self.A: InitA_Enum = InitA_Enum(kwargs.get('A'))
        self.E: InitE_Enum = InitE_Enum(kwargs.get('E'))
        self.D: NoneableString = kwargs.get('D', None)


class MainConfig_Output:
    def __init__(self, **kwargs):
        self.draw: str = kwargs.get('draw', False)
        self.normalization: bool = kwargs.get('normalization', False)
        self.sort: bool = kwargs.get('sort', False)
        self.metrics: NoneableString = kwargs.get('metrics', None)


class MainConfig_Params:
    def __init__(self, **kwargs):
        self.obj: NoneableString = kwargs.get('obj', None)
        self.around: List[float] = kwargs['around'] if kwargs['around'] else None


class MainCfg:
    def __init__(self, **kwargs):
        self.dataset: DatasetsEnum = DatasetsEnum(kwargs.get('dataset'))
        self.method: MethodsEnum = MethodsEnum(kwargs.get("method"))
        self.seed: int = kwargs.get('seed', 0)
        self.mode: ModeEnum = ModeEnum(kwargs.get('mode'))
        self.init: MainConfig_Init = MainConfig_Init(**kwargs['init']) if kwargs['init'] else None
        self.params: MainConfig_Params = MainConfig_Params(**kwargs['params']) if kwargs['params'] else None
        self.output: MainConfig_Output = MainConfig_Output(**kwargs['output']) if kwargs['output'] else None
