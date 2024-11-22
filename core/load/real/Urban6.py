from core.consts import REAL_DATASET_DIR, SIMULATED_DATASET_DIR
import numpy as np
import scipy.io as scio


def loader():
    P, L, N = 6, 162, 94249
    H, W = 307, 307

    data = scio.loadmat(f'{REAL_DATASET_DIR}/Urban/urban6.mat')
    Y = data['Y'].astype(np.float32)
    A = data['S_GT'].transpose((2, 0, 1)).reshape(P, N).astype(np.float32)
    E = data['GT'].T.astype(np.float32)
    return {
        "Y": Y,
        "A": A,
        "E": E,
        "P": P,
        "L": L,
        "N": N,
        "name": "Urban6",
        "H": H,
        "W": W
    }