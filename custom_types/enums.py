from enum import unique, IntEnum, auto


@unique
class DatasetsEnum(IntEnum):
    """ 数据集 """
    # 真实数据集
    Samson = auto()
    JasperRidge = auto()
    Urban4 = auto()
    Urban5 = auto()
    Urban6 = auto()
    # 模拟数据集
    DeepGun_ex2 = auto()
    DeepGun_ex4 = auto()
    SUnCNN_DC1 = auto()
    SUnCNN_DC2 = auto()


@unique
class MethodsEnum(IntEnum):
    """ 解混方法 """
    FCLSU = auto()
    SCLSU = auto()
    ELMM = auto()
    PGMSU = auto()
    Model5 = auto()
    CNNAEU = auto()
    UnDIP = auto()
    SUnCNN = auto()
    CyCU = auto()
    EGU = auto()


@unique
class InitE_Enum(IntEnum):
    """ 初始化endmember的方法 """
    GT = auto()
    VCA = auto()
    SiVM = auto()


@unique
class InitA_Enum(IntEnum):
    """ 初始化 abundanceMap 的方法 """
    GT = auto()
    FCLSU = auto()
    SCLSU = auto()
    SUnSAL = auto()


@unique
class HsiPropertyEnum(IntEnum):
    Y = auto()
    E = auto()
    A = auto()


@unique
class ModeEnum(IntEnum):
    Run = 1
    Param = 2
    Test = 3


@unique
class MetricsEnum(IntEnum):
    RMSE1 = auto()
    RMSE2 = auto()
    SAD1 = auto()
