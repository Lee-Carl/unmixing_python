from .FCLSU_INIT import FCLSU_INIT
from .SCLSU_INIT import SCLSU_INIT
from .SUnSAL_INIT import SUnSAL_INIT


class InitA:
    @staticmethod
    def FCLSU(data, S0):
        return FCLSU_INIT(data, S0)

    @staticmethod
    def SCLSU(data, S0):
        return SCLSU_INIT(data, S0)

    @staticmethod
    def SUnSAL(y, M, AL_iters=1000, lambda_0=0., positivity=False, addone=False, tol=1e-4, x0=None, verbose=False):
        return SUnSAL_INIT(y, M, AL_iters, lambda_0, positivity, addone, tol, x0, verbose)
