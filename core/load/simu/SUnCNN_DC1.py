from .Anchor import SIMULATED_DATASET_DIR
import numpy as np
import scipy.io as scio


def loader():
    P, L, N = 240, 224, 5625
    sqrtN, H, W = 75, 75, 75

    data = scio.loadmat(f'{SIMULATED_DATASET_DIR}/0db/DC1/Y_clean.mat')
    Y = data['Y_clean'].astype(np.float32)
    Y = Y.reshape(-1, L).T
    data = scio.loadmat(f'{SIMULATED_DATASET_DIR}/0db/DC1/XT.mat')
    A = data['XT'].astype(np.float32)
    A = A.reshape(P, -1)
    data = scio.loadmat(f'{SIMULATED_DATASET_DIR}/0db/DC1/EE.mat')
    E = data['EE'].astype(np.float32)
    return {
        "Y": Y,
        "A": A,
        "E": E,
        "P": P,
        "L": L,
        "sqrtN": sqrtN,
        "N": N,
        "name": "SUnCNN_DC1",
        "H": H,
        "W": W
    }
