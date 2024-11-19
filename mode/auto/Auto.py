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


# note:对已经收录的方法进行的比较
autometrics = AutoMetrics()  # 设置一套计算指标的方法
autodraw = AutoDraw()


class Auto:
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

    def computed(self, case, model):
        # 录入excel做准备
        file = self.get_PredDataDir(model=model, case=case)
        autometrics(case=case, file=file)

    def computed_all(self, ex):
        cases, models = ex['datasets'], ex['methods']
        # 录入excel做准备
        mp = []
        for case in cases:
            mp.append([case, "SAD", "E_aSAD", "RMSE", "A_aRMSE", "SAD_Y", "RMSE_Y", "aRMSE2"])
            for id, model in enumerate(models):
                if model == 'true':
                    continue
                print('*' * 100)
                print(f'case:{case}')
                print(f'model:{model}')
                file = self.get_PredDataDir(model=model, case=case)
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

    def plots_onepic_edm(self, ex, show=False):
        cases, models, names = ex['datasets'], ex['methods'], ex['edm_name']
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['font.family'] = 'Times New Roman'  # 字体
        font_prop = FontProperties(family='Times New Roman', style='normal', weight='bold', size=14)  # 加粗
        nums = len(models)  # 定义方法数

        for case in cases:  # 数据集数
            # 导出必要参数
            hsi = loadhsi(case)  # 数据集数据
            P = hsi["P"]  # 端元数
            L = hsi["L"]  # 端元数
            etrue = hsi['E']
            i = -1
            fig, axs = plt.subplots(P, nums, figsize=(16, 9))
            plt.subplots_adjust(wspace=0.5, hspace=0.5)
            nlist = names[case]  # 找到端元名称
            # 行数 -
            # 列数 -
            for m in models:
                i += 1

                # 将丰度转换为3维
                edm = self.get_PredData(model=m, case=case)

                # 真实端元图
                e_2d = edm['E'] if len(edm["E"].shape) == 2 else edm["E"][:, :, 0]
                norm = Norm()
                e_2d = norm.max_norm(e_2d)  # 归一化数据

                for j in range(0, P):
                    if j == 0:
                        # 第一行上面的方法名称
                        axs[j, i].title.set_text(m)  # 给左上角的子图添加标题
                        axs[j, i].title.set_fontproperties(font_prop)

                    if i == 0:
                        # 在轴外部添加文本
                        # axs[j, 0].text(-1.0, 0.8, 'EM #1', fontsize=14, rotation=90, va='top', ha='center')
                        # 第一列左侧侧的端元名称
                        if nlist:
                            # axs[j, 0].set_ylabel(nlist[j]).set_fontproperties(font_prop)
                            axs[j, 0].set_ylabel(nlist[j])
                    axs[j, i].plot(etrue[:, j], color="#FF0000", linestyle='-')
                    axs[j, i].plot(e_2d[:, j], color="#0000FF", linestyle='-.')
                    # 横轴
                    axs[j, i].set_xlim(0, L)
                    axs[j, i].set_xticks([_ for _ in range(0, L, 50)])
                    axs[j, i].set_xlabel("Bands")
                    # 纵轴
                    axs[j, i].set_ylim(0, 1)
                    axs[j, i].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])  # 设置y轴的刻度标签（不包括0.5，如果您不需要这个值）
                    axs[j, i].set_ylabel("Reflectance")
            # plt.tight_layout()  # 紧凑布局
            # 添加图例
            axs[P - 1, nums // 2].legend(bbox_to_anchor=(0.5, -0.1), ncol=2, fancybox=True,
                                         shadow=True)

            if show:
                plt.show()

    def plots_onepic_abu(self, ex, show=False, todiff=False, t=False):
        cases, models, names = ex['datasets'], ex['methods'], ex['edm_name']
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['font.family'] = 'Times New Roman'  # 字体
        font_prop = FontProperties(family='Times New Roman', style='normal', weight='bold', size=14)  # 加粗
        if not todiff:
            # 如果是丰度图，不是插值图
            models.insert(0, "GT")
        nums = len(models)  # 定义方法数

        for case in cases:  # 数据集数
            # 导出必要参数
            hsi = loadhsi(case)  # 数据集数据
            P = hsi["P"]  # 端元数
            H = hsi["H"]  # 端元数
            W = hsi["W"]  # 端元数
            i = -1
            fig, axs = plt.subplots(P, nums, figsize=(12, 9))
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            nlist = names[case]  # 找到端元名称
            # 行数 -
            # 列数 -
            for m in models:
                i += 1

                # 将丰度转换为3维
                if m == "GT":
                    abu = hsi
                else:
                    abu = self.get_PredData(model=m, case=case)

                if not todiff:
                    # 真实丰度图
                    a_3d = abu['A'].reshape(P, H, W)
                else:
                    # 差值图
                    a_2d = np.fabs(abu['A'] - hsi['A'])
                    a_3d = a_2d.reshape(P, H, W)

                for j in range(0, P):
                    # 方法名称
                    if j == 0:
                        axs[j, i].title.set_text(m)  # 给左上角的子图添加标题
                        # axs[j, i].set_xlabel("123")  # 给左上角的子图添加标题
                        axs[j, i].title.set_fontproperties(font_prop)
                    # 端元名称
                    if i == 0:
                        if nlist:
                            axs[j, 0].set_ylabel(nlist[j]).set_fontproperties(font_prop)

                    # plt.subplot(P, nums, i + nums * j)
                    norm = Normalize(vmin=0, vmax=0.4)  # 这里的vmin和vmax根据您的数据自行设置
                    im = a_3d[j] if not t else a_3d[j].T
                    axs[j, i].imshow(im, cmap='jet', interpolation='none', norm=norm)
                    # if i == nums - 1 and j == P - 1:  # 仅在最后一个位置加入colorbar
                    #     cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.6])
                    #     fig.colorbar(a, cax=cbar_ax)
                    axs[j, i].set_xticks([])
                    axs[j, i].set_yticks([])
            # plt.tight_layout() # 紧凑布局
            # colorbar
            cbar_ax = fig.add_axes([0.93, 0.15, 0.025, 0.68])
            colorbar = fig.colorbar(axs[0, 0].images[-1], cax=cbar_ax)
            # colorbar_labels
            # ticks = [[i / 10 for i in range(0, 10 + 1, 1)]]
            # colorbar.set_ticks([i / 100 for i in range(0, 30 + 1, 10)])
            # colorbar.set_ticklabels([str(i / 100.0) for i in range(0, 30 + 1, 10)], fontsize=14)
            # colorbar.set_label('Abundance', rotation=270, labelpad=20)
            if show:
                plt.show()

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

    def sort_all(self, ex):
        cases, models = ex['datasets'], ex['methods']
        # 作用: 对所有计算结果进行排序
        for case in cases:
            for model in models:
                data_true = loadhsi(case)

                data_true['Y'] = Norm.max_norm(data_true['Y'])

                savepos = self.get_PredDataDir(model=model, case=case)

                dpred = sio.loadmat(savepos)

                # dp = DataProcessor(data_true)
                # dpred = dp.sort_EndmembersAndAbundances(data_true, dpred)

                data_pred = dpred['A']
                # print(data_pred.keys(),data_pred['P'])
                P, H, W, N = data_true['P'], data_true['H'], data_true['W'], data_true['N']
                data_pred = data_pred.reshape(P, H, W)
                data_pred = data_pred.transpose(0, 2, 1)
                data_pred = data_pred.reshape(P, N)
                dpred['A'] = data_pred

                sio.savemat(savepos, dpred)

                print(f'{case}-{model}-{savepos}:完成')
