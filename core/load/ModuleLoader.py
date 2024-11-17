import os
import yaml
from typing import Any
from utils import ImportUtil
from core import consts


class ModuleLoader:
    @staticmethod
    def get_Metrics_Function(way: str):
        try:
            with open(consts.RESULTS_METRICS_CONFIG_FILE, 'r') as file:
                data = yaml.safe_load(file)  # 读取配置文件信息
                return ImportUtil.loadModule(data[way])  # 导出模块
        except Exception as e:
            print(f"An exception occurred:{e}")

    @staticmethod
    def get_Draw_Function(way: str):
        try:
            with open(consts.RESULTS_DRAW_CONFIG_FILE, 'r') as file:
                data = yaml.safe_load(file)  # 读取配置文件信息
                return ImportUtil.loadModule(data[way])  # 导出模块
        except Exception as e:
            print(f"An exception occurred:{e}")

    @staticmethod
    def get_Init_Function(way: str):
        try:
            with open(consts.RESULTS_INIT_CONFIG_FILE, 'r') as file:
                data = yaml.safe_load(file)  # 读取配置文件信息
                return ImportUtil.loadModule(data[way])  # 导出模块
        except Exception as e:
            print(f"An exception occurred:{e}")

    @staticmethod
    def get_Method(method_name: str):
        filename = f'{consts.METHODS_CONFIG_DIR}/{method_name}.yaml'
        with open(filename, 'r') as file:
            yaml_data = yaml.safe_load(file)
            return ImportUtil.loadModule(yaml_data['src'])

    @staticmethod
    def get_Method_params(dataset_name: str, method_name: str) -> dict:
        filename = f'{consts.METHODS_CONFIG_DIR}/{method_name}.yaml'
        with open(filename, 'r') as file:
            yaml_data = yaml.safe_load(file)
        params_data = yaml_data['params']
        return params_data[dataset_name] if dataset_name in params_data.keys() else params_data['default']
