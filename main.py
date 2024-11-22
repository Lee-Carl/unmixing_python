from mode.Unmixing import Unmixing
from custom_types import DatasetsEnum, ModeEnum, MethodsEnum, InitA_Enum, InitE_Enum

around1 = [0, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1]
around2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
around3 = [i * 1.0 / 1e5 for i in range(1, 10 + 1, 1)]
around = around2

unmxingInfo = {
    # 数据集
    "dataset": DatasetsEnum.Urban4,
    # 方法
    "method": MethodsEnum.PGMSU,
    # 模式
    "mode": ModeEnum.Run,
    # 随机种子
    "seed": 0,
    # 初始化方式
    "init": {
        "custom_init_data": None,
        "custom_init_method": None,
        "snr": 0,
        "normalization": True,
        "E": InitE_Enum.VCA,
        "A": InitA_Enum.SUnSAL,
        "D": None,
        "show_initdata": True,
    },
    # 输出
    "output": {
        "sort": True,
        "draw": 'default',
        "metrics": 'default',
    },
    # 调参模式的配置
    "params": {
        "obj": "lambda_kl",
        "around": around
    }
}

if __name__ == '__main__':
    u = Unmixing(unmxingInfo)
    u()
