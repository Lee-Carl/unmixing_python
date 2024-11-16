import functools
import numpy as np


def shapecheck(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 提取所有参数
        all_args = args + tuple(kwargs.values())

        # 检查前两个参数是否为numpy数组并且形状是否相同
        if len(all_args) >= 2:
            first_arg, second_arg = all_args[:2]
            if isinstance(first_arg, np.ndarray) and isinstance(second_arg, np.ndarray):
                if first_arg.shape != second_arg.shape:
                    raise ValueError("The shapes of the first two arguments do not match.")

        return func(*args, **kwargs)

    return wrapper


@shapecheck
def some_function(*args, **kwargs):
    # 函数实现
    print("Function called with args:", args, "and kwargs:", kwargs)


# 正确的函数调用方式
some_function(1, arr1=2)  # 位置参数在前，关键字参数在后
some_function(arr1=np.array([1, 2]), arr2=np.array([1, 2]))
