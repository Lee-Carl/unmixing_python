import numpy as np
from core.wraps import checkShape


@checkShape
def SRE(data_true, data_pred):
    sre = 20 * np.log10(np.linalg.norm(data_true, ord=2) / np.linalg.norm(data_true - data_pred, ord=2))
    return sre
