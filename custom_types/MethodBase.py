import copy
from abc import abstractmethod, ABC
from typing import Any, Dict
from .HsiDataset import HsiDataset


class MethodBase(ABC):
    def __init__(self, params: dict, init: HsiDataset) -> None:
        if not isinstance(params, dict):
            raise TypeError(f"params must be an instance of dict instead of {type(params)}")
        if not isinstance(init, HsiDataset):
            raise TypeError(f"init must be an instance of HsiDataset instead of {type(init)}")
        self.params = params
        self.init = init

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Dict:
        pass
