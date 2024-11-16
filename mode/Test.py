import os
from .run import RunMode
from .params import ParamsMode
from core import DataProcessor, ModeAdapter
import scipy.io as sio

# 获取父目录的父目录的绝对地址,即Study2的绝对地址
current_dir = os.path.dirname(os.path.abspath(__file__))
anchor = os.path.abspath(os.path.join(current_dir, ".."))
anchor = anchor.replace('\\', '/')
# 主要的配置文件
MAIN_DIR = anchor

# 定义类
rm = RunMode()
pm = ParamsMode()


class TestMode:
    def __init__(self):
        pass

    @staticmethod
    def get_Data(abspath=None, datasetName=None, relpath=None):
        if relpath:
            dtrue_name = [part for part in relpath.split("/") if part != ""]
            dtrue_name = dtrue_name[2]
            dtrue = ip.loadhsi(dtrue_name)
            file = f'{MAIN_DIR}/{relpath}/results.mat'
            dpred = sio.loadmat(file)
            return dtrue, dpred
        elif abspath and datasetName:
            dtrue = ip.loadhsi(datasetName)
            dpred = sio.loadmat(abspath)
            return dtrue, dpred
        else:
            print("请提供(方式一)相对地址，或(方式二)绝对地址和数据集名称")
            exit(0)

    @staticmethod
    def get_Abspath_ByRelpath(relpath):
        return f'{MAIN_DIR} / {relpath}/'

    def analysis_params(self, abspath=None, relpath=None):
        if abspath:
            route = abspath
        else:
            route = self.get_Abspath_ByRelpath(relpath)

        # 分析调参程序
        pm.analysis_params(route)

    @staticmethod
    def test_initdata(replace=False):
        cp = ModeAdapter()
        cp.set_seed()
        dataset = cp.get_Dataset()
        init = cp.get_InitData(dataset, replace=replace)
        rm.test_initData(dataset, init)

    def draw(self, abspath=None, datasetName=None, relpath=None):
        dtrue, dpred = self.get_Data(abspath=abspath, datasetName=datasetName, relpath=relpath)

        un = InitProcessor()
        dtrue['Y'] = un.normalization(dtrue['Y'])
        rm.get_Pictures(dtrue, dpred)

    def compute(self, abspath=None, datasetName=None, relpath=None):
        data_true, dpred = self.get_Data(abspath=abspath, datasetName=datasetName, relpath=relpath)

        un = InitProcessor()
        data_true['Y'] = un.normalization(data_true['Y'])

        sm = rm.get_Metrics(data_true, dpred)
        print(sm.__str__())
        dd = rm.get_Pictures(data_true, dpred)
        dd()

        # epred2 = extract_edm(data_true['Y'], data_pred['A'])
        # print(sm.compute_SAD(data_true['E'], epred2))

    def changeshape(self, abspath=None, datasetName=None, relpath=None):
        data_true, data_pred = self.get_Data(abspath=abspath, datasetName=datasetName, relpath=relpath)

        un = InitProcessor()
        data_true['Y'] = un.normalization(data_true['Y'])

        dp = DataProcessor(data_true)

        data_pred = dp.sort_EndmembersAndAbundances(data_true, data_pred)
        P, H, W = data_true['P'], data_true['H'], data_true['W']

        abu3d = data_pred['A'].reshape(P, H, W)
        for i, abu in enumerate(abu3d):
            abu3d[i, :, :] = abu.T
        data_pred['A'] = abu3d.reshape(P, -1)
