import functools
from typing import Callable, Any


def checkShape(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        all_args = args + tuple(kwargs.values())  # 提取所有参数
        if len(all_args) >= 2:  # 检查前两个参数是否为numpy数组并且形状是否相同
            first_arg, second_arg = all_args[:2]
            if first_arg.shape != second_arg.shape:
                raise ValueError("The shapes of the first two arguments do not match.")

        return func(*args, **kwargs)

    return wrapper
