from .hyperVca import hyperVca
from .sivm import SiVM


class InitE:
    @staticmethod
    def VCA(M, q):
        return hyperVca(M, q)

    @staticmethod
    def SiVM(Y, p, seed=0, *args, **kwargs):
        return SiVM(Y, p, seed, *args, **kwargs)
