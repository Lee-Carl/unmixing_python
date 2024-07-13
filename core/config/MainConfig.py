import os
import numpy as np
import scipy.io as sio
import yaml
from .Anchor import MAIN_CONFIG_FILE, MAIN_CONFIG_DIR, DATA_DIR


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


class MainConfig:
    @staticmethod
    def get():
        with open(MAIN_CONFIG_FILE, 'r') as file:
            yaml_data = yaml.safe_load(file)
        return yaml_data

    @remove_after_loadmat
    def get_InitData(self, dataset_name, init_str):
        """
        params:
            dataset_name: the name of the dataset
            init_str: the specific methods of init expressed in a way of str
        return:
            flag: True represents that fine the results
            data: represents the data. if the value of it is 0 , it means that cannot find it
        """
        # 获取缓存文件
        filename = f'{DATA_DIR}/{dataset_name}/{init_str}.mat'
        # 首先，从注册在yaml中的数据（数据一般在当前项目中）进行查找
        if os.path.exists(filename):
            data = sio.loadmat(filename)
            print("初始化数据: 正载入缓存...")
            return True, data
        else:
            # 查找不到的结果
            return False, 0
