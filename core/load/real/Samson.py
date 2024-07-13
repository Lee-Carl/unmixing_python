from .Anchor import REAL_DATASET_DIR
import numpy as np
import scipy.io as scio


def loader():
    P, L, N = 3, 156, 9025
    H, W = 95, 95

    data = scio.loadmat(f'{REAL_DATASET_DIR}/Samson/hu-ae-main/Samson.mat')
    Y = data['Y'].astype(np.float32)

    Y = Y.reshape(L, H, W)
    for i, a in enumerate(Y):
        Y[i, :, :] = a.T
    Y = Y.reshape(L, N)

    A = data['S_GT'].transpose((2, 0, 1)).reshape(P, N).astype(np.float32)
    E = data['GT'].T.astype(np.float32)
    return {
        "name": "Samson",
        "Y": Y,
        "A": A,
        "E": E,
        "P": P,
        "L": L,
        "N": N,
        "H": H,
        "W": W
    }
