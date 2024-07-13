from .Anchor import SIMULATED_DATASET_DIR
import numpy as np
import scipy.io as scio


def loader():
    P, L, N = 5, 162, 2500
    H, W = 50, 50

    data = scio.loadmat(f'{SIMULATED_DATASET_DIR}/20db/data_ex4.mat')
    Y = data['r']
    Y = Y.astype(np.float32)
    M = data['M']
    Mvs = data['Mvs']
    A = data['alphas']
    A = A.astype(np.float32)

    return {
        "Y": Y,
        "E": M,
        "E_3d": Mvs,
        "A": A,
        "P": P,
        "L": L,
        "N": N,
        "name": "DeepGUn_ex4",
        "H": H,
        "W": W
    }
