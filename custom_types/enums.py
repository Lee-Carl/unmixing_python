from enum import unique, IntEnum, auto


class IntBaseEnum(IntEnum):
    @classmethod
    def get_name(cls, key: int):
        for member in cls:
            if member.value == key:
                return member.name
        raise ValueError(f"{cls.__name__}没有{key}属性")

    @classmethod
    def showProperty(cls):
        print("-" * 50)
        print(f"{cls.__name__}的成员变量有:")
        for member in cls:
            print(f"{member.value}.{member.name}")
        print("-" * 50)


@unique
class DatasetsEnum(IntBaseEnum):
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
class MethodsEnum(IntBaseEnum):
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
class InitE_Enum(IntBaseEnum):
    """ 初始化endmember的方法 """
    GT = auto()
    VCA = auto()
    SiVM = auto()


@unique
class InitA_Enum(IntBaseEnum):
    """ 初始化 abundanceMap 的方法 """
    GT = auto()
    FCLSU = auto()
    SCLSU = auto()
    SUnSAL = auto()


@unique
class HsiPropertyEnum(IntBaseEnum):
    Y = auto()
    E = auto()
    A = auto()


@unique
class ModeEnum(IntBaseEnum):
    Run = 1
    Param = 2
    Test = 3


@unique
class MetricsEnum(IntBaseEnum):
    RMSE1 = auto()
    RMSE2 = auto()
    SAD1 = auto()
