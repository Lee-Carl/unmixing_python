from core.consts import REAL_DATASET_DIR, SIMULATED_DATASET_DIR
import numpy as np
import scipy.io as scio


def loader():
    P, L, N = 5, 162, 2500
    H, W = 50, 50
    path: str = f'{SIMULATED_DATASET_DIR}/30db/data_ex4.mat'
    data = scio.loadmat(path)
    Y = data['r']
    Y = Y.astype(np.float32)
    M = data['M']
    Mvs = data['Mvs']
    A = data['alphas']
    A = A.astype(np.float32)

    return {
        "Y": Y,
        "E": M,
        "A": A,
        "P": P,
        "L": L,
        "N": N,
        "name": "DeepGUn_ex4",
        "H": H,
        "W": W,
        "other": {
            "E_3d": Mvs,
        }
    }
