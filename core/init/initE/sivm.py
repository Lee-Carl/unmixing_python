import numpy as np


def Eucli_dist(x, y):
    a = np.subtract(x, y)
    return np.dot(a.T, a)


def SiVM(Y, p, seed=0, *args, **kwargs):
    x, p = Y, p

    [D, N] = x.shape
    # If no distf given, use Euclidean distance function
    Z1 = np.zeros((1, 1))
    O1 = np.ones((1, 1))
    # Find farthest point
    d = np.zeros((p, N))
    index = np.zeros((p, 1))
    V = np.zeros((1, N))
    ZD = np.zeros((D, 1))
    for i in range(N):
        d[0, i] = Eucli_dist(x[:, i].reshape(D, 1), ZD)

    index = np.argmax(d[0, :])

    for i in range(N):
        d[0, i] = Eucli_dist(x[:, i].reshape(D, 1), x[:, index].reshape(D, 1))

    for v in range(1, p):
        D1 = np.concatenate(
            (d[0:v, index].reshape((v, index.size)), np.ones((v, 1))), axis=1
        )
        D2 = np.concatenate((np.ones((1, v)), Z1), axis=1)
        D4 = np.concatenate((D1, D2), axis=0)
        D4 = np.linalg.inv(D4)

        for i in range(N):
            D3 = np.concatenate((d[0:v, i].reshape((v, 1)), O1), axis=0)
            V[0, i] = np.dot(np.dot(D3.T, D4), D3)

        index = np.append(index, np.argmax(V))
        for i in range(N):
            d[v, i] = Eucli_dist(
                x[:, i].reshape(D, 1), x[:, index[v]].reshape(D, 1)
            )

    per = np.argsort(index)
    index = np.sort(index)
    d = d[per, :]
    E = x[:, index]
    return E
