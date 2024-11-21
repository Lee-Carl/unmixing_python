from .enums import DatasetsEnum, MethodsEnum


class ExInfo:
    def __init__(self, dataset: int, method: int, src: str, dst: str):
        self.dataset = DatasetsEnum(dataset)
        self.method = MethodsEnum(method)
        self.src = src
        self.dst = dst
