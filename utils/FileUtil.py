import os
from typing import Union
import yaml
import scipy.io as sio


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
