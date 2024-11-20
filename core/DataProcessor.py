from custom_types import HsiDataset, HsiData, InitE_Enum, InitA_Enum, DatasetsEnum
from .init import InitEdm, InitAbu, Norm, Noise, set_pytorch_seed
from .func import hsiSort
from .load import loadhsi, loadInitData
import numpy as np


class DataProcessor:
    @staticmethod
    def loadDatast(case: DatasetsEnum) -> HsiDataset:
        return loadhsi(case.name)

    @staticmethod
    def set_seed(seed: int = 0):
        set_pytorch_seed(seed)

    @classmethod
    def gen_initData(cls, data: HsiDataset, initE: InitE_Enum, initA: InitA_Enum, initD: int = None,
                     snr: float = 0, normalization: bool = True, seed: int = 0) -> HsiDataset:
        Y_init: HsiData = data.pixels.copy()
        Y_init = cls.addNoise(Y_init, snr)
        Y_init = cls.norm(Y_init, normalization)
        E_init = cls.gen_endmembers(Y_init=Y_init, dataset=data, initE=initE, seed=seed)
        A_init = cls.gen_abundances(Y_init=Y_init, E_init=E_init, dataset=data, initA=initA, seed=seed)
        init_dic = {
            'Y': Y_init,
            'E': E_init,
            'A': A_init,
            'D': np.zeros(0),
            'P': data.P,
            'L': data.L,
            'N': data.N,
            'H': data.H,
            'W': data.W,
            'name': f'{str(snr)}db_{initE.name}_{initA.name}',
            'other': {
                "src": data.name
            }
        }
        init: HsiDataset = HsiDataset(**init_dic)
        return init

    @staticmethod
    def gen_endmembers(Y_init: HsiData, dataset: HsiDataset, initE: InitE_Enum, seed=0) -> HsiData:
        data: HsiData
        if initE == InitE_Enum.GT:
            data = dataset.E.astype(np.float32).copy()
        elif initE == InitE_Enum.SiVM:
            data = InitEdm.SiVM(Y_init.copy(), dataset.P, seed=seed)  # 通过VCA生成端元
        else:
            data, _, _ = InitEdm.VCA(Y_init.copy(), dataset.P)  # 通过VCA生成端元
        return data

    @staticmethod
    def gen_abundances(Y_init: HsiData, E_init: HsiData, dataset: HsiDataset, initA: InitA_Enum,
                       seed=0) -> HsiData:
        data: HsiData
        if initA == InitA_Enum.GT:
            data = dataset.abu.astype(np.float32).copy()
        elif initA == InitA_Enum.FCLSU:
            data = InitAbu.FCLSU(Y_init, E_init)
        elif initA == InitA_Enum.SCLSU:
            data = InitAbu.SCLSU(Y_init, E_init)
        elif initA == InitA_Enum.SUnSAL:
            data = InitAbu.SUnSAL(Y_init, E_init)[0]
        else:
            raise ValueError("initA:Unknown Methods")
        return data

    @staticmethod
    def addNoise(data: HsiData, snr: float) -> HsiData:
        if snr > 0:
            n = Noise()
            data = n.noise2(data, snr)
        return data

    @staticmethod
    def norm(data, normalization=True) -> HsiData:
        if normalization:
            n = Norm()
            data = n.max_norm(data)
        return data

    @staticmethod
    def gen_Y(e, a):
        # 作用：用端元矩阵与丰度矩阵生成像元矩阵Y
        # 丰度一般是二维的
        # 端元有二维，也有三维的。假设E2=(198,4),E3=（198，4，10**4），A=（4，10**4）
        E, A = e.copy(), a.copy()
        if A.shape[0] > A.shape[1]:
            A = A.T  # A=(4,10**4)

        if len(E.shape) == 2:
            if E.shape[0] < E.shape[1]:
                E = E.T  # E2=(198,4)
            return E @ A  # y=(198,10**4)
        else:
            A = A.T  # A=(10**4,4)
            A = np.expand_dims(A, axis=1)  # A=(10**4,1,4)
            s = E.shape
            s = sorted(enumerate(s), key=lambda x: x[1])  # e=[(1, 4), (0, 198), (2, 10000)]
            E = E.transpose(s[2][0], s[0][0], s[1][0])  # E3=(10**4,4,198)
            Y = A @ E  # y=(10**4,1,198)
            Y = np.squeeze(Y, axis=1)  # y=(10**4,198)
            Y = Y.T  # y=(198,10**4)
            return Y

    @staticmethod
    def sort_edm(dtrue: HsiData, dpred: HsiData, P, func=None, repeat: bool = False):
        hsiSort.sort_edm(dtrue, dpred, P, func=None, repeat=False)

    @staticmethod
    def sort_abu(dtrue: HsiData, dpred: HsiData, P, func=None, repeat: bool = False):
        hsiSort.sort_abu(dtrue, dpred, P, func=None, repeat=False)

    @staticmethod
    def sort_edm_and_abu(dtrue: HsiDataset, dpred: HsiDataset, case: int = 2,
                         repeat: bool = False, edm_repeat: bool = False,
                         abu_repeat: bool = False, tip: bool = False)->HsiDataset:
        return hsiSort.sort_Edm_And_Abu(dtrue, dpred, case, repeat,
                                        edm_repeat, abu_repeat, tip)

    @classmethod
    def oneClickProc(cls, data: HsiData, snr=0, normalization=True) -> HsiData:
        data = cls.addNoise(data, snr)
        data = cls.norm(data, normalization)
        return data

    @staticmethod
    def getInitData(dataset_name: str, init_str: str):
        return loadInitData(dataset_name, init_str)
