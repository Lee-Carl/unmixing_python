import os
import scipy.io as sio
from core import consts
from core.wraps import remove_after_loadmat


@remove_after_loadmat
def loadInitData(dataset_name: str, init_str: str):
    """
    params:
        dataset_name: the name of the dataset
        init_str: the specific methods of init expressed in a way of str
    return:
        flag: True represents that fine the results
        data: represents the data. if the value of it is 0 , it means that cannot find it
    """
    # 获取缓存文件
    filename = f'{consts.INITDATA_DIR}/{dataset_name}/{init_str}.mat'
    # 首先，从注册在yaml中的数据（数据一般在当前项目中）进行查找
    if os.path.exists(filename):
        data = sio.loadmat(filename)
        print("初始化数据: 正载入缓存...")
        return True, data
    else:
        # 查找不到的结果
        return False, 0
