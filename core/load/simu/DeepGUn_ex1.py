from .Anchor import SIMULATED_DATASET_DIR
import numpy as np
import scipy.io as scio


def loader():
    P, L, N = 3, 224, 4900
    H, W = 70, 70
    data = scio.loadmat(f'{SIMULATED_DATASET_DIR}/20db/data_ex1.mat')
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
        "name": "DeepGUn_ex1",
        "H": H,
        "W": W
    }
