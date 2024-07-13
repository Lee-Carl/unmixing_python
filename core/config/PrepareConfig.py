import os
import importlib
import yaml
from .Anchor import RESULTS_METRICS_CONFIG_FILE, RESULTS_DRAW_CONFIG_FILE, RESULTS_INIT_CONFIG_FILE


class PrepareConfig:
    @staticmethod
    def __importlib_module(module_name):
        try:
            module, attr = os.path.splitext(module_name)
            module = importlib.import_module(module)
            model = getattr(module, attr[1:])
            return model
        except BaseException as b:
            print(f"An exception occurred: {b}")

    def get_Metrics_Function(self, way):
        try:
            with open(RESULTS_METRICS_CONFIG_FILE, 'r') as file:
                data = yaml.safe_load(file)  # 读取配置文件信息
                return self.__importlib_module(data[way])  # 导出模块
        except Exception as e:
            print(f"An exception occurred:{e}")

    def get_Draw_Function(self, way):
        try:
            with open(RESULTS_DRAW_CONFIG_FILE, 'r') as file:
                data = yaml.safe_load(file)  # 读取配置文件信息
                return self.__importlib_module(data[way])  # 导出模块
        except Exception as e:
            print(f"An exception occurred:{e}")

    def get_Init_Function(self, way):
        try:
            with open(RESULTS_INIT_CONFIG_FILE, 'r') as file:
                data = yaml.safe_load(file)  # 读取配置文件信息
                return self.__importlib_module(data[way])  # 导出模块
        except Exception as e:
            print(f"An exception occurred:{e}")
