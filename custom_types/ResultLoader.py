from core.consts import RESULTS_DIR
from core import DataProcessor
from custom_types import DatasetsEnum, MethodsEnum

dp = DataProcessor()


class ResultLoader:
    def __init__(self, dataset: DatasetsEnum, method: MethodsEnum, idx: int = 0, path: str = ''):
        self.dataset = dataset
        self.method = method
        self.idx = idx
        self.path = path

    @staticmethod
    def get_Data(abspath=None, datasetName=None, relpath=None):
        if relpath:
            dtrue_name = [part for part in relpath.split("/") if part != ""]
            dtrue_name = dtrue_name[2]
            dtrue = dp.loadDatast(dtrue_name)
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
