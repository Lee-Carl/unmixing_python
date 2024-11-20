import math
import xlsxwriter as xw
import shutil
import os
from .metrics import AutoMetrics
from .draw import AutoDraw
from custom_types import DatasetsEnum, MethodsEnum
from utils import FileUtil
from core import Analyzer, consts
from typing import List, Union, TypeVar, Generic, Any

# note:对已经收录的方法进行的比较
autometrics = AutoMetrics()  # 设置一套计算指标的方法
autodraw = AutoDraw()
ana = Analyzer()


class AutoModeInfo:
    def __init__(self, dataset: int, method: int, src: str, dst: str):
        self.dataset = DatasetsEnum(dataset)
        self.method = MethodsEnum(method)
        self.src = src
        self.dst = dst


class AutoMode:
    def __init__(self, params: dict, ex: dict):
        self.src = params['src']
        self.dst = params['dst']
        self.draw = params['draw']
        self.xlsx = params['xlsx']
        self.infos: List[List[AutoModeInfo]] = []
        self.ex = ex
        self.baseDir = consts.RESULTS_DIR
        self.results: List[List[Any]] = []

    def __call__(self):
        # 1. 复制结果到指定目录
        self.copyToNewPath(self.src, self.dst)
        # 2. 自动计算与画图
        self.compute()
        self.draw()

    def copyToNewPath(self, src: str, dst: str):
        self.infos = []
        for i, ds in enumerate(self.ex["datasets"]):
            collect: List[AutoModeInfo] = []
            for method in self.ex["methods"]:
                ''' 判断路径是否存在 '''
                dir1: str = os.path.join(self.baseDir, ds, method)
                if os.path.exists(dir1):
                    raise ValueError(f"不存在名称为{dir1}的目录")
                if not os.path.isdir(dir1):
                    continue
                records: List[str] = []
                ''' 若存在，则遍历目录 '''
                for record in os.listdir(dir1):
                    dir2: str = os.path.join(dir1, record)
                    ''' 如果存在default目录，则直接将此目录视为目标目录，并直接终止此循环 '''
                    if record == 'default':
                        records.clear()
                        records.append(dir2)
                        break
                    ''' 是目录；非空；不以params开头 '''
                    if os.path.isdir(dir2) and not FileUtil.is_directory_empty(dir2) and \
                            not record.startswith(consts.RESULTS_PARAMS_DIR_PREFIX):
                        records.append(dir2)
                if records:
                    src_file: str = FileUtil.get_latest_directory(records)  # 源目录绝对地址
                    dst_file: str = src_file.replace(src, dst)  # 目标目录绝对地址
                    collect.append(AutoModeInfo(method, ds, src_file, dst_file))
                    shutil.copytree(src_file, dst_file)  # copytree会创建虚拟的目录树，从而能在目标目录不存在时完成复制
            self.infos.append(collect)

    @staticmethod
    def checkNan(data):
        # 将Nan数据全部置0
        if math.isnan(data):
            data = 0
        return data

    def compute(self):
        # 录入excel做准备
        self.results = []
        for items in self.infos:
            for item in items:
                # todo: 待完善
                self.results.append(
                    [item.method.value, "SAD", "E_aSAD", "RMSE", "A_aRMSE", "SAD_Y", "RMSE_Y", "aRMSE2"])
                az = Analyzer(dataset=item.dataset, method=item.method, path=item.dst)
                data: Any = az.call_any_function(az.getDataset, az.getDataset())
                data = self.checkNan(data)
                self.results.append([])
        self.save()

    def save(self):
        if self.xlsx:
            with xw.Workbook(f"{self.dst}/vs.xlsx") as workbook:  # 创建工作簿
                worksheet1 = workbook.add_worksheet("对比数据")  # 创建子表
                worksheet1.activate()  # 激活表
                # 设置单元格居中格式
                cell_format = workbook.add_format()
                cell_format.set_align('center')
                cell_format.set_align('vcenter')
                for i, info in enumerate(self.results):
                    worksheet1.write_row('A' + str(i + 1), info, cell_format)
                worksheet1.set_column('A:H', 15)
                worksheet1.set_column('B:B', 70)
                worksheet1.set_column('D:D', 70)
        print('*' * 100)

    def plot(self):
        # 录入excel做准备
        for items in self.infos:
            for item in items:
                # todo: 待完善
                az = Analyzer(dataset=item.dataset, method=item.method, path=item.dst)
                data: Any = az.call_any_function(az.getDataset, az.getDataset())
                data = self.checkNan(data)
                self.results.append([])
