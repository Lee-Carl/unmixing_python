import os
from typing import Union
import yaml
import scipy.io as sio
import datetime
import shutil


class FileUtil:
    @staticmethod
    def createdir(dn):
        if not os.path.exists(dn):
            os.makedirs(dn)

    @staticmethod
    def getAbsPath_ByRelativepath(path: str) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        anchor = os.path.abspath(os.path.join(current_dir, path))
        anchor = anchor.replace('\\', '/')
        return anchor

    @staticmethod
    def writeFile(fpath: str, text: str, mode='w') -> None:
        file = open(fpath, mode)
        file.write(text)
        file.close()

    @staticmethod
    def writeYamlFile(fpath: str, data: dict, mode: str = 'w', default_flow_style: bool = False):
        with open(fpath, mode) as file:
            yaml.dump(data, file, default_flow_style=default_flow_style)

    @staticmethod
    def readYamlFile(fpath: str, data: dict, mode: str = 'w', default_flow_style: bool = False):
        pass

    @staticmethod
    def savemat(fpath: str, data: dict):
        sio.savemat(fpath, data)

    @staticmethod
    def is_directory_empty(directory):
        # 获取目录下的所有文件和子目录名称
        files_and_directories = os.listdir(directory)

        # 检查列表是否为空
        if len(files_and_directories) == 0:
            return True
        else:
            return False

    @staticmethod
    def get_subdirectories(directory):
        subdirectories = []
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                subdirectories.append(item_path)
        return subdirectories

    @staticmethod
    def get_latest_directory(directories):
        # 获取所有目录的创建时间
        dir_ctimes = [(directory, datetime.datetime.fromtimestamp(os.path.getctime(directory))) for directory in
                      directories]
        # 找到创建时间最新的目录
        latest_dir = sorted(dir_ctimes, key=lambda x: x[1], reverse=True)[0]
        return latest_dir[0]
