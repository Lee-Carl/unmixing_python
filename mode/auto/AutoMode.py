import math

import matplotlib.pyplot as plt
import xlsxwriter as xw
import shutil
import os

import scipy.io as sio
import core.draw as draw
from core.load import loadhsi
from core.SmartMetrics import SmartMetrics
from .metrics import AutoMetrics
from .draw import AutoDraw
import numpy as np
from core.init import Norm
from matplotlib.font_manager import FontProperties
from matplotlib.colors import Normalize
from utils import FileUtil
from core import Analyzer

# note:对已经收录的方法进行的比较
autometrics = AutoMetrics()  # 设置一套计算指标的方法
autodraw = AutoDraw()
ana = Analyzer()


class AutoMode:
    def __init__(self, params):
        self.obj_file = params['obj_file']
        self.src = params['src']
        self.dst = params['dst']
        self.draw = params['draw']
        self.xlsx = params['xlsx']

    def get_PredData(self, model, case):
        dst = self.dst
        pred_dir = os.path.join(dst, model, case)
        latest_dir = FileUtil.get_latest_directory(FileUtil.get_subdirectories(pred_dir))
        results_file = os.path.join(latest_dir, self.obj_file)
        data_pred = sio.loadmat(results_file)
        return data_pred

    def computed_all(self, ex):
        cases, models = ex['datasets'], ex['methods']
        # 录入excel做准备
        mp = []
        for case in cases:
            mp.append([case, "SAD", "E_aSAD", "RMSE", "A_aRMSE", "SAD_Y", "RMSE_Y", "aRMSE2"])
            for id, model in enumerate(models):
                SAD, aSAD, rmse, aRMSE, SAD_Y, RMSE_Y, aRMSE2 = autometrics(case=case, file=file)
                aSAD = self.checkNan(aSAD)
                aRMSE = self.checkNan(aRMSE)
                SAD_Y = self.checkNan(SAD_Y)
                RMSE_Y = self.checkNan(RMSE_Y)
                aRMSE2 = self.checkNan(aRMSE2)
                mp.append([model, str(SAD), aSAD, str(rmse), aRMSE, SAD_Y, RMSE_Y, aRMSE2])
            mp.append([])

        if self.xlsx:
            with xw.Workbook(f"{self.dst}/vs.xlsx") as workbook:  # 创建工作簿
                worksheet1 = workbook.add_worksheet("对比数据")  # 创建子表
                worksheet1.activate()  # 激活表
                # 设置单元格居中格式
                cell_format = workbook.add_format()
                cell_format.set_align('center')
                cell_format.set_align('vcenter')
                for i, info in enumerate(mp):
                    worksheet1.write_row('A' + str(i + 1), info, cell_format)
                worksheet1.set_column('A:H', 15)
                worksheet1.set_column('B:B', 70)
                worksheet1.set_column('D:D', 70)

        print('*' * 100)

    def plots_one(self, ex, types, show=False):
        cases, models = ex['datasets'], ex['methods']
        for case in cases:
            # 生成保存地址
            savepath = os.path.join(self.dst, self.draw, f'{case}')
            # self.createdir(os.path.join(os.path.join(self.dst, '_draw')))
            FileUtil.createdir(savepath)
            # 导出真实数据
            dtrue = loadhsi(case)
            dtrue['Y'] = Norm.max_norm(dtrue['Y'])
            am = SmartMetrics(dtrue)
            P, L, N = am.dp.getPLN()
            H, W = am.dp.getHW()
            for model in models:
                print(f'当前画的是：{model}--{case}')

                # 导出预测数据
                dpred = self.get_PredData(model=model, case=case)

                for t in types:
                    if t == "abu":
                        apred = dpred['A']
                        apred = am.dp.transpose(apred, (P, N))
                        apred = apred.reshape((P, H, W))
                        draw.abundanceMap(abu=apred, savepath=savepath, name=model, show=show)
                    elif t == "edm":
                        norm = Norm()
                        x = norm.max_norm(dpred['E'])
                        draw.vs_endmembers(dtrue['E'], x, name=model, savepath=savepath)
                    elif t == "abu_diff":
                        diff = np.fabs(dtrue["A"] - dpred['A'])
                        diff = diff.reshape(P, H, W)
                        draw.abundanceMap(abu=diff, savepath=savepath, name=model, show=show)

    def plots_all(self, ex, types, show=False):
        cases, models = ex['datasets'], ex['methods']
        for case in cases:
            # 生成保存地址
            savepath = os.path.join(self.dst, self.draw, f'{case}')
            # self.createdir(os.path.join(os.path.join(self.dst, '_draw')))
            FileUtil.createdir(savepath)
            # 导出真实数据
            dtrue = loadhsi(case)
            dtrue['Y'] = Norm.max_norm(dtrue['Y'])
            am = SmartMetrics(dtrue)
            P, L, N = am.dp.getPLN()
            H, W = am.dp.getHW()
            for model in models:
                print(f'当前画的是：{model}--{case}')

                # 导出预测数据
                dpred = self.get_PredData(model=model, case=case)

                for t in types:
                    if t == "abu":
                        apred = dpred['A']
                        apred = am.dp.transpose(apred, (P, N))
                        apred = apred.reshape((P, H, W))
                        draw.abundanceMap_all(abu=apred, savepath=savepath, show=show, name=model)
                    elif t == "edm":
                        norm = Norm()
                        x = norm.max_norm(dpred['E'])
                        draw.vs_endmembers_all(dtrue['E'], x, name=model, savepath=savepath)
                    elif t == "abu_diff":
                        diff = np.fabs(dtrue["A"] - dpred['A'])
                        diff = diff.reshape(P, H, W)
                        draw.abundanceMap_all(abu=diff, savepath=savepath, show=show, name=model)

    @staticmethod
    def checkNan(data):
        # 将Nan数据全部置0
        if math.isnan(data):
            data = 0
        return data

    def get_PredDataDir(self, model, case):
        pred_dir = os.path.join(self.dst, model, case)
        latest_dir = FileUtil.get_latest_directory(FileUtil.get_subdirectories(pred_dir))
        results_file = os.path.join(latest_dir, self.obj_file)
        return results_file

    def getLatestDirInfo(self):
        src = self.src
        dst = self.dst
        for ds in os.listdir(src):  # 遍历res目录
            dataset_dir = os.path.join(src, ds)  # 拼接成绝对地址
            if os.path.isdir(dataset_dir):
                for methods in os.listdir(dataset_dir):  # 遍历单一的数据集目录
                    methods_dir = os.path.join(src, ds, methods)
                    sub_dirs = []  # 收集方法目录下所有的目录文件
                    for record in os.listdir(methods_dir):  # 遍历单一的方法目录
                        record_dir = os.path.join(src, ds, methods, record)
                        # 如果存在default目录，则直接将此目录视为目标目录，并直接终止此循环
                        if record == 'default':
                            sub_dirs.clear()
                            sub_dirs.append(record_dir)
                            break
                        # 是目录；非空；不以params开头
                        if os.path.isdir(record_dir) and \
                                not FileUtil.is_directory_empty(record_dir) and \
                                not record.startswith('params'):
                            sub_dirs.append(record_dir)
                    if sub_dirs:
                        src_file = FileUtil.get_latest_directory(sub_dirs)  # 源目录绝对地址
                        dst_file = src_file.replace(src, dst)  # 目标目录绝对地址
                        shutil.copytree(src_file, dst_file)  # copytree会创建虚拟的目录树，从而能在目标目录不存在时完成复制
