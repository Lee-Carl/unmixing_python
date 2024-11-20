from core.consts import RESULTS_DIR
from core import DataProcessor, consts
from custom_types import DatasetsEnum, MethodsEnum, HsiDataset, ModeEnum
from typing import Union
import scipy.io as sio

from utils import FileUtil

dp = DataProcessor()


class ResultLoader:

    def __init__(self,
                 dataset: Union[DatasetsEnum, None] = None,
                 method: Union[MethodsEnum, None] = None,
                 mode: ModeEnum = ModeEnum.Run,
                 idx: int = 1, path: str = ''):
        self.datasetName = dataset
        self.method = method
        self.idx = idx
        self.path = path
        self.data = self.get_ResultData(dataset=dataset, method=method, mode=mode, idx=idx, path=path)
        self.dataset = dp.loadDatast(dataset)

    @staticmethod
    def get_ResultData(dataset: DatasetsEnum, method: MethodsEnum, mode: ModeEnum, idx: int = 1,
                       path: str = '') -> HsiDataset:
        filepath: str = ''
        if path:
            filepath = f'{path}/{consts.RESULTS_FILE}'
        elif dataset and method:
            pre: str = consts.RESULTS_RUN_DIR_PREFIX if mode == ModeEnum.Run else consts.RESULTS_PARAMS_DIR_PREFIX
            filepath = f'{method.name}/{dataset.name}/{pre}{idx}/{consts.RESULTS_FILE}'
        if len(filepath) == 0:
            raise ValueError("请检查传递的参数")
        absPath: str = FileUtil.getAbsPath_ByRelativepath(filepath)
        data: dict = sio.loadmat(absPath)
        return HsiDataset(**data)
