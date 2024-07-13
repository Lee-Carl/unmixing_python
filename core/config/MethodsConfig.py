import os
import importlib
import yaml
from .Anchor import METHODS_CONFIG_DIR


class MethodsConfig:
    @staticmethod
    def __importlib_module(module_name):
        try:
            module, attr = os.path.splitext(module_name)
            module = importlib.import_module(module)
            model = getattr(module, attr[1:])
            return model
        except BaseException as b:
            print(f"An exception occurred: {b}")

    def get_Method(self, method_name):
        filename = f'{METHODS_CONFIG_DIR}/{method_name}.yaml'
        with open(filename, 'r') as file:
            yaml_data = yaml.safe_load(file)
            return self.__importlib_module(yaml_data['src'])

    @staticmethod
    def get_Method_params(dataset_name: str, method_name: str):
        filename = f'{METHODS_CONFIG_DIR}/{method_name}.yaml'
        with open(filename, 'r') as file:
            yaml_data = yaml.safe_load(file)
        params_data = yaml_data['params']
        return params_data[dataset_name] if dataset_name in params_data.keys() else params_data['default']
