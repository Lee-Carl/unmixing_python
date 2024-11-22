from core.consts import REAL_DATASET_DIR, SIMULATED_DATASET_DIR
import numpy as np
import scipy.io as scio


def loader():
    P, L, N = 4, 198, 10000
    H, W = 100, 100
    data = scio.loadmat(f'{REAL_DATASET_DIR}/JasperRidge/JasperRidge2_R198.mat')

    Y = data['Y']  # (C,w*h)
    # Y = np.reshape(Y, [198, 100, 100])
    # for i, y in enumerate(Y):
    #     Y[i] = y.T
    # Y = np.reshape(Y, [198, 10000])
    Y = Y.astype(np.float32)

    A = scio.loadmat(f'{REAL_DATASET_DIR}/JasperRidge/JasperRidge2_end4.mat')['A']
    # 为了方便画丰度图，所以需要转置一下丰度，如果这样做，Y也需要转置
    # 写法一:
    # A = np.reshape(A, (4, 100, 100))
    # for i, a in enumerate(A):
    #     A[i] = a.T
    # A = np.reshape(A, (4, 10000))

    # 写法二:
    # A = A.reshape(P, H, W)
    # A = A.transpose(0, 2, 1)
    # A = A.reshape(P, N)

    A = A.astype(np.float32)

    E = scio.loadmat(f'{REAL_DATASET_DIR}/JasperRidge/JasperRidge2_end4.mat')['M']

    return {
        "Y": Y,
        "E": E,
        "A": A,
        "P": P,
        "L": L,
        "N": N,
        "name": "JasperRidge",
        "H": H,
        "W": W
    }
