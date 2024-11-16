import importlib
import os
from .Anchor import DATA_LOADER_CONFIG
from custom_types import HsiDataset
import yaml


def __importlib_module(module_name):
    """
    导入并返回一个路径格式的模块对象。

    参数：
        module_name(str): 一个包含模块名称的字符串，该模块名称应具有以下格式：
            "package.module"
            或
            "package.module.ClassName"。

    返回：
        object: 成功时返回与module_name对应的类或模块对象；出错时返回None。

    异常：
        如果module_name格式不正确或者无法导入相应的模块或类，将捕获异常并打印错误信息。"""
    try:
        module, attr = os.path.splitext(module_name)
        module = importlib.import_module(module)
        model = getattr(module, attr[1:])
        return model
    except BaseException as b:
        print(f"An exception occurred: {b}")


def loadhsi(case: str) -> HsiDataset:
    # 读取data_loader.yaml中的数据
    with open(DATA_LOADER_CONFIG, 'r', encoding='utf-8') as file:
        config_data = yaml.safe_load(file)

    if case in config_data.keys():
        # 如果存在相应的数据集, 那么导入
        d = __importlib_module(config_data[case])
        return HsiDataset(**d())
    else:
        # 不存在相应的数据集, 就报错
        raise "There is no such case in the dataset_loader.yaml"
