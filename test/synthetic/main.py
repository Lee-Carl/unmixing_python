import scipy.io as sio
from core.load import loadlib
from core.init import Noise

def fun():
    noi = Noise()
    data = loadlib("USGS_1995")
    D = data['datalib']
    E = D[:, 1:5]
    SNR = 20
    A = 0
    # N = noi.noise1(Y,)
    Y = E * A + N
    P = A.shape[0]
    L = Y.shape[0]
    N = Y.shape[1]
    H = 50
    W = 50
    dataset = {
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
    print(data.shape)


if __name__ == '__main__':
    fun()
