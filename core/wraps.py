import functools
from typing import Callable, Any
import numpy as np


# 用来去除初始化数据中整数总是带有"[[]]"的问题
def remove_after_loadmat(func):
    def wrapper(*args, **kwargs):
        flag, data = func(*args, **kwargs)
        # flag如果为true，后面跟的数据需要进行处理
        if flag:
            data_key1 = ['P', 'L', 'N', 'H', 'W']
            data_key2 = ['Y', 'A', 'E', 'D']
            # 对keys的数据进行降维
            for key in data_key1:
                if key in data.keys() and np.ndim(data[key]) >= 1:
                    data[key] = data[key].item()
            # 对data_key转换数据类型
            for key in data_key2:
                if key in data.keys() and isinstance(data[key], np.ndarray):
                    data[key] = data[key].astype(np.float32)
        return flag, data

    return wrapper


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
