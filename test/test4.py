import copy
from enum import IntEnum
from typing import Union

from typing_extensions import get_args, TypeAlias


class ABC:
    def __init__(self):
        self.a = 123
        self.aaa = {
            'b': 123,
            'c': "fd"
        }

    @property
    def num(self):
        return self.a

    @num.setter
    def num(self, val):
        self.a = val

    def copy(self):
        return copy.deepcopy(self)


class TestEnum(IntEnum):
    a = 1
    b = 2


def int_to_enum(value, enum_class):
    for item in enum_class:
        if item.value == value:
            return item
    raise ValueError(f"{value} is not a valid enum value for {enum_class.__name__}")


etype: TypeAlias = Union[int, float]
print(get_args(etype))
a = 1
print(isinstance(a, get_args(etype)))

b = '1'
print(isinstance(b, get_args(etype)))
