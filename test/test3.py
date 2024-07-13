import torch
import core.draw as draw
from core.load import loadhsi
import scipy.io as sio
import numpy as np


def loadtrue():
    data = loadhsi("mgf")
    d = data['A'].reshape(data['P'], data['H'], data['W'])
    draw.abundanceMap_all(d, savepath="./")


def loadpred():
    case = 'JasperRidge'
    # pred
    r1 = f"../_exp/ex1/PGMSU/{case}/0"
    dpred = sio.loadmat(f"{r1}/results.mat")
    # true
    dtrue = loadhsi(case)
    P, L, N, H, W = dtrue["P"], dtrue["L"], dtrue["N"], dtrue["H"], dtrue["W"]
    # diff
    # draw.vs_endmembers(dtrue["E"], dpred['E'], savepath=r1)
    # diff = np.fabs(dtrue["A"] - dpred['A'])
    # diff = diff.reshape(P, H, W)
    y = dtrue['Y']

    em = dpred['E']
    em = em.transpose(2, 1, 0)
    draw.scatter_map(y=y, em_hat=em, p=dtrue['P'])


if __name__ == '__main__':
    # loadtrue()
    loadpred()
