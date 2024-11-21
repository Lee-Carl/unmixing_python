from core.consts import RESULTS_DIR
from core import DataProcessor, consts
from custom_types import DatasetsEnum, MethodsEnum, HsiDataset, ModeEnum
from typing import Union
import scipy.io as sio
import numpy as np
from utils import FileUtil

dp = DataProcessor()


class Analyzer:

    def __init__(self,
                 dataset: Union[DatasetsEnum, None] = None,
                 method: Union[MethodsEnum, None] = None,
                 mode: ModeEnum = ModeEnum.Run,
                 idx: int = 1, path: str = ''):
        self.datasetName = dataset
        self.method = method
        self.idx = idx
        self.path = path
        self.savePath = self.get_SavePath(dataset=dataset, method=method, mode=mode, idx=idx, path=path)
        self.data = self.get_ResultData(self.savePath)
        self.dataset = dp.loadDatast(dataset)

    @staticmethod
    def get_SavePath(dataset: DatasetsEnum, method: MethodsEnum, mode: ModeEnum, idx: int = 1,
                     path: str = '') -> str:
        filepath: str = ''
        if path:
            filepath = f'{path}/{consts.RESULTS_FILE}'
        elif dataset and method:
            pre: str = consts.RESULTS_RUN_DIR_PREFIX if mode == ModeEnum.Run else consts.RESULTS_PARAMS_DIR_PREFIX
            filepath = f'{method.name}/{dataset.name}/{pre}{idx}/{consts.RESULTS_FILE}'
        if len(filepath) == 0:
            raise ValueError("请检查传递的参数")
        absPath: str = FileUtil.getAbsPath_ByRelativepath(filepath)
        return absPath

    @staticmethod
    def get_ResultData(savePath: str) -> HsiDataset:
        data: dict = sio.loadmat(savePath)
        return HsiDataset(**data)

    def sort(self) -> None:
        # 作用: 对所有计算结果进行排序
        self.data = dp.sort_edm_and_abu(dtrue=self.dataset, dpred=self.data, case=1, repeat=False)

    def save(self):
        sio.savemat(self.savePath, self.data.__dict__)

    @staticmethod
    def call_any_function(func, args):
        pass

    def getDataset(self):
        return self.dataset.copy()

    def getKey(self, key: str):
        return self.dataset[key], self.data[key]

    def get_differenceMap(self, shapePHW: bool = False):
        data = np.fabs(self.dataset.abu - self.data.abu)
        if shapePHW:
            P: int = self.dataset.P
            H: int = self.dataset.H
            W: int = self.dataset.W
            return data.reshape(P, H, W)
        else:
            return data

    def get_abundanceMap(self, shapePHW: bool = False):
        data = self.dataset.abu.copy()
        if shapePHW:
            P: int = self.dataset.P
            H: int = self.dataset.H
            W: int = self.dataset.W
            return data.reshape(P, H, W)
        else:
            return data
