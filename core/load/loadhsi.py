import importlib
import os
from core import consts
from custom_types import HsiDataset, DatasetsEnum
from utils import ImportUtil
import yaml


def loadhsi(case: str) -> HsiDataset:
    # 读取data_loader.yaml中的数据
    with open(consts.DATA_LOADER_CONFIG, 'r', encoding='utf-8') as file:
        config_data = yaml.safe_load(file)

    if case in config_data.keys():
        # 如果存在相应的数据集, 那么导入
        ds = config_data[case]
        d = ImportUtil.loadModule(ds)
        return HsiDataset(**d())
    else:
        # 不存在相应的数据集, 就报错
        raise "There is no such case in the dataset_loader.yaml"
