from mode import Unmixing
from custom_types import DatasetsEnum, ModeEnum, MethodsEnum, InitA_Enum, InitE_Enum
from core.load import loadhsi

# around1 = [0, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1]
# around2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
# around = around2
#
# unmxingInfo = {
#     "dataset": DatasetsEnum.JasperRidge,
#     "methods": MethodsEnum.PGMSU,
#     "mode": ModeEnum.Run,
#     "seed": 0,
#     "init": {
#         "custom_init_data": None,
#         "custom_init_method": None,
#         "snr": 0,
#         "normalization": True,
#         "E": InitE_Enum.VCA,
#         "A": InitA_Enum.SUnSAL,
#         "D": None,
#         "show_initdata": None,
#     },
#     "output": {
#         "sort": True,
#         "draw": 'default',
#         "metrics": 'default',
#     },
#     "params": {
#         "obj": "lambda_kl",
#         "around": around
#     }
# }

if __name__ == '__main__':
    u = Unmixing()
    u()
