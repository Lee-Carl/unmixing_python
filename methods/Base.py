from abc import abstractmethod
from typing import Any, Dict


class Base:
    def __init__(self, params, init):
        self.params = params
        self.init = init

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Dict:
        pass
