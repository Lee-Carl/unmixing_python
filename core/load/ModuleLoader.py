import os
import yaml
from typing import Any
from utils import ImportUtil

# 获取父目录的父目录的绝对地址,即Study2的绝对地址
current_dir = os.path.dirname(os.path.abspath(__file__))
anchor = os.path.abspath(os.path.join(current_dir, "..", ".."))
anchor = anchor.replace('\\', '/')
# 文件/目录的绝对地址
MAIN_CONFIG_FILE = f'{anchor}/config/main_config.yaml'
MAIN_CONFIG_DIR = f'{anchor}/config/'
DATA_DIR = f'{anchor}/data/initData/'
METHODS_CONFIG_DIR = f'{anchor}/config/methods/'
RESULTS_DRAW_CONFIG_FILE = f'{anchor}/config/results/draw.yaml'
RESULTS_METRICS_CONFIG_FILE = f'{anchor}/config/results/metrics.yaml'
RESULTS_INIT_CONFIG_FILE = f'{anchor}/config/prepare/init.yaml'


class ModuleLoader:
    @staticmethod
    def get_Metrics_Function(way):
        try:
            with open(RESULTS_METRICS_CONFIG_FILE, 'r') as file:
                data = yaml.safe_load(file)  # 读取配置文件信息
                return ImportUtil.loadModule(data[way])  # 导出模块
        except Exception as e:
            print(f"An exception occurred:{e}")

    @staticmethod
    def get_Draw_Function(way):
        try:
            with open(RESULTS_DRAW_CONFIG_FILE, 'r') as file:
                data = yaml.safe_load(file)  # 读取配置文件信息
                return ImportUtil.loadModule(data[way])  # 导出模块
        except Exception as e:
            print(f"An exception occurred:{e}")

    @staticmethod
    def get_Init_Function(way):
        try:
            with open(RESULTS_INIT_CONFIG_FILE, 'r') as file:
                data = yaml.safe_load(file)  # 读取配置文件信息
                return ImportUtil.loadModule(data[way])  # 导出模块
        except Exception as e:
            print(f"An exception occurred:{e}")

    @staticmethod
    def get_Method(method_name):
        filename = f'{METHODS_CONFIG_DIR}/{method_name}.yaml'
        with open(filename, 'r') as file:
            yaml_data = yaml.safe_load(file)
            return ImportUtil.loadModule(yaml_data['src'])

    @staticmethod
    def get_Method_params(dataset_name: str, method_name: str):
        filename = f'{METHODS_CONFIG_DIR}/{method_name}.yaml'
        with open(filename, 'r') as file:
            yaml_data = yaml.safe_load(file)
        params_data = yaml_data['params']
        return params_data[dataset_name] if dataset_name in params_data.keys() else params_data['default']
