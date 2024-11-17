import os
import importlib
from typing import Any


class ImportUtil:
    @staticmethod
    def loadModule(module_name: str) -> Any:
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
