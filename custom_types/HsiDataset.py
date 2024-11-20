from typing import Union, Dict, Any
import numpy as np
import copy
from utils.TypeUtil import TypeUtil
from .type_alias import HsiData

class HsiDataset(object):
    def __init__(self, **data: Dict[str, Dict[str, Any]]):
        assert TypeUtil.is_HsiData(data['Y']), f"数据集中像元类型有误为{type(data['Y'])}"
        assert TypeUtil.is_HsiData(data['E']), f"数据集中端元类型有误为{type(data['E'])}"
        assert TypeUtil.is_HsiData(data['A']), f"数据集中丰度类型有误为{type(data['A'])}"
        # assert isinstance(data["D"], HsiData.__args__), f"数据集中光谱库类型有误为{type(data['D'])}"
        assert isinstance(data.get("L"), int) and data.get("L"), "数据集中端元数异常"
        assert isinstance(data.get("P"), int) and data.get("P"), "数据集中波段数异常"
        assert isinstance(data.get("N"), int) and data.get("N"), "数据集中像素点数异常"
        assert isinstance(data.get("H"), int) and data.get("H"), "数据集中图像高度值异常"
        assert isinstance(data.get("W"), int) and data.get("W"), "数据集中图像宽度值异常"
        self.Y: Union = data.get("Y")
        self.E: Union = data.get("E")
        self.A: Union = data.get("A")
        self.D: Union = data.get("D")
        #
        self.L: int = data.get("L", 0)
        self.P: int = data.get("P", 0)
        self.N: int = data.get("N", 0)
        self.H: int = data.get("H", 0)
        self.W: int = data.get("W", 0)
        self.name: Union[str, None] = data.get("name", None)
        self.other: dict = data.get("other", None)

    def getPLN(self):
        return self.P, self.L, self.N

    def getHW(self):
        return self.H, self.W

    @property
    def pixels(self):
        return self.Y

    @pixels.setter
    def pixels(self, val: HsiData):
        self.Y = val

    @property
    def edm(self):
        # endmembers
        return self.E

    @edm.setter
    def edm(self, val: HsiData):
        self.E = val

    @property
    def abu(self):
        # abundances
        return self.A

    @abu.setter
    def abu(self, val: HsiData):
        self.A = val

    @property
    def pixelNum(self):
        return self.N

    @property
    def bandNum(self):
        return self.L

    @property
    def edmNum(self):
        return self.P

    @property
    def imgWidth(self):
        return self.W

    @property
    def imgHeight(self):
        return self.H

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def copy(self):
        return copy.deepcopy(self)
