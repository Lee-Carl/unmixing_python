from ..load import loadhsi as loadds
from ..init import InitEdm, InitAbu, Norm, Noise, set_pytorch_seed
from .DataProcessor import DataProcessor
import scipy.io as sio
import numpy as np
import os
import torch
import random


class InitProcessor:

    @staticmethod
    def set_seed(seed):
        set_pytorch_seed(seed)

    @staticmethod
    def loadhsi(name):
        return loadds(name)  # load the dataset

    @staticmethod
    def loadlib(name):
        return loadds(name)  # load the datalib

    @staticmethod
    def addNoise(data, snr):
        if snr > 0:
            n = Noise()
            data = n.noise2(data, snr)
        return data

    @staticmethod
    def normalization(data, normalization=True):
        if normalization:
            n = Norm()
            data = n.max_norm(data)
        return data

    def copeWithData(self, data, snr=0, normalization=True):
        data = self.addNoise(data, snr)
        data = self.normalization(data, normalization)
        return data

    def generateInitData(self, data: dict, initE: str, initA: str, initD: str = None, savepath: str = None, snr=0,
                         normalization=True, seed=0):
        P, L, N = data['P'], data['L'], data['N']
        H, W = data['H'], data['W']

        Y_init = self.copeWithData(data['Y'].copy(), snr, normalization)

        if initE == 'GT':
            E_init = data['E'].astype(np.float32).copy()
        elif initE == 'SiVM':
            E_init = InitEdm.SiVM(Y_init.copy(), P, seed=seed)  # 通过VCA生成端元
        else:
            E_init, _, _ = InitEdm.VCA(Y_init.copy(), P)  # 通过VCA生成端元

        if initA == 'GT':
            A_init = data['A'].astype(np.float32).copy()
        elif initA == 'FCLSU':
            A_init = InitAbu.FCLSU(Y_init, E_init)
        elif initA == 'SCLSU':
            A_init = InitAbu.SCLSU(Y_init, E_init)
        elif initA == 'SUnSAL':
            A_init = InitAbu.SUnSAL(Y_init, E_init)[0]
        else:
            raise ValueError("initA:Unknown Methods")

        init_str = f'{str(snr)}db_{initE}_{initA}'
        init = {
            'Y': Y_init,
            'E': E_init,
            'A': A_init,
            'P': P,
            'L': L,
            'N': N,
            'H': H,
            'W': W,
            'name': init_str,
            "src": data["name"]
        }

        # 为端元和丰度排序
        dp = DataProcessor(data)
        init = dp.sort_EndmembersAndAbundances(data, init, repeat=True, case=1, tip=False)

        if savepath:
            print(savepath)
            filepos = savepath + f'/{init_str}.mat'
            sio.savemat(filepos, init)

        return init
