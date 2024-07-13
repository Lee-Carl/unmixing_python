from .Anchor import SIMULATED_DATASET_DIR
import numpy as np
import scipy.io as scio


def loader():
    P, L, N = 5, 221, 16384
    H, W = 128, 128

    data = scio.loadmat(f'{SIMULATED_DATASET_DIR}/20db/MaternGF.mat')
    Y = data['Y'].astype(np.float32)
    A = data['A'].astype(np.float32)
    E = data['E'].astype(np.float32)

    return {
        "Y": Y,
        "A": A,
        "E": E,
        "P": P,
        "L": L,
        "N": N,
        "name": "MaternGF",
        "H": H,
        "W": W
    }
