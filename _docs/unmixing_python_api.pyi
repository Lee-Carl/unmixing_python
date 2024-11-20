from custom_types import HsiDataset, HsiData, DatasetsEnum, InitA_Enum, InitE_Enum, MainCfg, HsiPropertyEnum

''' core.func '''

def extract_edm(y: HsiData, a:HsiData)-> HsiData: ...
"""
    提取端元
    Args:
        y: 像元数据
        a: 预测的丰度数据

    Returns:
        预测的端元数据
"""

def sort_Edm_And_Abu(dtrue: HsiDataset, dpred: HsiDataset, case: int = 2, repeat: bool = False,
                     edm_repeat: bool = False, abu_repeat: bool = False,
                     tip: bool = False): ...
"""
    对端元和丰度排序，使两者一致

    Args:
        dtrue: 像元数据
        dpred: 预测的丰度数据
        case: 使用哪一种方式排序
        repeat: 是否允许端元+丰度重复, true允许
        edm_repeat: 是否允许端元+丰度重复, true允许
        abu_repeat: 是否允许端元+丰度重复, true允许
        tip:

    Returns:
        排序后的数据
"""  

def __get_similarity_matrix(true_, pred_, P, similarity_func, tip=None): ...


def __choose_similarity_max(similarity_matrix, P, repeat, tip=None): ...


def __choose_similarity_min(similarity_matrix, P, repeat, tip=None): ...


def __sort_framework(true_, pred_, P, func, repeat): ...


def sort_edm(dtrue, dpred, P, func=None, repeat=False): ...
"""
    对端元排序

    Args:
        dtrue: 像元数据
        dpred: 预测的丰度数据
        P: 端元数
        func: 排序函数。如果不填写自定义的排序函数，会使用默认的排序函数
        repeat: 是否允许重复

    Returns:
        排序好的端元数据
"""

def sort_abu(dtrue, dpred, P, func=None, repeat=False): ...
"""
    对丰度排序

    Args:
        dtrue: 像元数据
        dpred: 预测的丰度数据
        P: 端元数
        func: 排序函数。如果不填写自定义的排序函数，会使用默认的排序函数
        repeat: 是否允许重复

    Returns:
        排序好的丰度数据
"""

''' core.load '''


def loadhsi(case: str) -> HsiDataset: ...
"""
    导入数据集

    Args:
        case: 数据集名称。使用时，可以手写数据集名称，但必须与config.settings.data_loader.yaml中的数据集名称一致;也可以通过DatasetsEnum，并使用get_name()实现

    Returns:
        包装成HsiDataset的数据集
"""

def loadInitData(dataset_name: str, init_str: str): ...


class ModuleLoader:
    @staticmethod
    def get_Metrics_Function(way: str): ...
    """
        Args:
            way: 指标函数名

        Returns:
            指标函数
    """

    @staticmethod
    def get_Draw_Function(way: str): ...
    """
        Args:
            way: 函数名

        Returns:
            用matplotlib库实现的函数
    """

    @staticmethod
    def get_Init_Function(way: str): ...
    """
        Args:
            way: 初始化函数名

        Returns:
            初始化函数
    """

    @staticmethod
    def get_Method(method_name: str): ...
    """
        Args:
            way: 解混方法名

        Returns:
            解混方法
    """

    @staticmethod
    def get_Method_params(dataset_name: str, method_name: str) -> dict: ...
    """
        Args:
            way: 指标函数名

        Returns:
            指标函数
    """


''' core '''


class DataProcessor:
    @staticmethod
    def loadDatast(case: DatasetsEnum) -> HsiDataset: ...
    """
        Args:
            case: 数据集相关的枚举变量

        Returns:
            数据集
    """

    @staticmethod
    def set_seed(seed: int = 0)->None: ...
    """
        固定随机种子。 固定的是pytorch + numpy + random(py自带)库的随机种子

        Args:
            seed: 随机种子
    """

    @classmethod
    def gen_initData(cls, data: HsiDataset, initE: InitE_Enum, initA: InitA_Enum, initD: int = None,
                     snr: float = 0, normalization: bool = True, seed: int = 0) -> HsiDataset: ...
    """
        生成端元和，丰度数据

        Args:
            data: 数据集
            initE: 初始化端元的方式
            initA: 初始化丰度的方式
            initD: 未实现的功能，不用传此参数。
            snr: 噪声
            normalization: 是否要正则化
            seed: 随机种子

        Returns:
           包装成HsiDataset的初始化数据
    """

    @staticmethod
    def gen_endmembers(Y_init: HsiData, dataset: HsiDataset, initE: InitE_Enum, seed=0) -> HsiData: ...
    """
        生成端元数据

        Args:
            Y_init: 混元数据
            dataset: 数据集
            initE: 初始化端元的方式
            seed: 随机种子

        Returns:
            生成的端元数据
    """

    @staticmethod
    def gen_abundances(Y_init: HsiData, E_init: HsiData, dataset: HsiDataset, initA: InitA_Enum,
                       seed=0) -> HsiData: ...
    """
        生成丰度数据

        Args:
            Y_init: 混元数据
            dataset: 数据集
            E_init: 初始化端元的方式
            initA: 初始化丰度的方法
            seed: 随机种子

        Returns:
            生成的端元数据
    """

    @staticmethod
    def addNoise(data: HsiData, snr: float) -> HsiData: ...
    """
        增加噪声

        Args:
            data: 数据
            snr: 噪声(db)

        Returns:
            增加过噪声的数据
    """

    @staticmethod
    def norm(data, normalization=True) -> HsiData: ...
    """
        归一化数据

        Args:
            data: 数据

        Returns:
            归一化后的数据
    """

    @staticmethod
    def gen_Y(e, a): ...
    """
        通过端元和丰度合成像元数据

        Args:
            e: 端元数据
            a: 丰度数据

        Returns:
            合成的像元数据
    """

    @staticmethod
    def sort_edm(dtrue: HsiData, dpred: HsiData, P, func=None, repeat: bool = False): ...
    """
        对端元进行排序

        Args:
            dtrue: 真实的端元数据
            dpred: 预测的端元数据
            P: 端元数
            func: (选填)排序函数。
            repeat: (选填)允许重复。

        Returns:
            排序后的端元数据
    """

    @staticmethod
    def sort_abu(dtrue: HsiData, dpred: HsiData, P, func=None, repeat: bool = False): ...
    """
        对丰度进行排序

        Args:
            dtrue: 真实的丰度数据
            dpred: 预测的丰度数据
            P: 端元数
            func: (选填)排序函数。
            repeat: (选填)允许重复。

        Returns:
            排序后的丰度数据
    """

    @staticmethod
    def sort_edm_and_abu(dtrue: HsiDataset, dpred: HsiDataset, case: int = 2,
                         repeat: bool = False, edm_repeat: bool = False,
                         abu_repeat: bool = False, tip: bool = False): ...
    """
        对丰度和端元进行排序

        Args:
            dtrue: 真实的丰度数据集
            dpred: 包装成HsiDataset的预测数据
            case: (选填)排序方法，可选值有: 1，2
            P: 端元数
            func: (选填)排序函数。
            repeat: (选填)允许重复。
            edm_repeat: (选填)允许端元重复。
            abu_repeat: (选填)允许丰度重复。
            tip: (选填)提示。

        Returns:
            排序后的端元、丰度数据
    """

    @classmethod
    def oneClickProc(cls, data: HsiData, snr=0, normalization=True) -> HsiData: ...
    """
        一键处理数据(增加噪声、归一化)

        Args:
            data: 数据
            snr: 分贝
            normalization: 是否归一化

        Returns:
            处理后的数据
    """

    @staticmethod
    def getInitData(dataset_name: str, init_str: str): ...
    """
        得到初始化数据

        Args:
            dataset_name: 数据名称
            init_str: 初始化的方式

        Returns:
            初始化数据
    """


class ModeAdapter:
    def __init__(self, cfg: MainCfg): ...

    def set_seed(self, seed: int = 0): ...
    """
        固定随机种子
    """

    def __getInitStr(self) -> str: ...

    def get_Dataset(self) -> HsiDataset: ...
    """
        得到数据集
    """

    def get_InitData(self, dataset: HsiDataset, replace=False) -> HsiDataset: ...
    """
        得到初始化数据
    """

    def get_Model(self): ...
    """
        得到解混方法
    """

    def get_params(self) -> dict: ...
    

    @staticmethod
    def run(method, params: dict, init: HsiDataset, savepath=None, output_display=True) -> HsiDataset: ...
    """
        运行模型
        
        Args:
            method: 解混方法
            params: 解混参数，对应config.methods目录下的文件。
            init: 包装成HsiDataset的初始化数据集
            savepath: (选填)存储路径
            output_display: (选填)是否显示进度条

        Returns:
            包装成HsiDataset的预测数据集
    """

    def get_Params_adjust(self): ...
    """
        得到调整的参数，对应配置表的"params"信息
    """

    @staticmethod
    def sort_EndmembersAndAbundances(dataset: HsiDataset, datapred: HsiDataset) -> HsiDataset: ...
    """
        对端元和丰度进行排序
        运行模型
        
        Args:
            dataset: 真实数据集
            datapred: 包装成HsiDataset的预测数据

        Returns:
            排序后的预测数据
    """

    def compute(self, dataset, datapred, out_path=None)->None: ...
    """
        调用自定义的指标类
    """

    def draw(self, dataset, datapred, out_path=None)->None: ...
    """
        调用自定义的绘画类
    """


class MyLog: # 日志类
    def __init__(self, cfg: MainCfg): ...

    def __genOutDir(self) -> str: ...

    def get_outdir(self): ...

    @classmethod
    def __dict_recursive(cls, dictionary, indent=0): ...

    def record(self, out_path): ...

    @staticmethod
    def record_inyaml(content, outpath): ...


class SmartMetrics: # 计算指标，在原本的基础上增加了对数据的形状判断
    def __init__(self, dataset: HsiDataset, tip=True) -> None: ...

    def compute_RMSE(self, A_true, A_pred): ...
    """
        计算RMSE(详情见CYCU的RMSE)
    """

    def compute_RMSE_2(self, A_true, A_pred): ...
    """
        计算RMSE(详情见PGMSU的RMSE)
    """

    def compute_RMSE_a2(self, A_true, A_pred): ...
    """
        计算RMSE
    """

    def compute_SAD_3D(self, data_true, data_pred): ...
    """
        计算SAD, 用于处理这种情况: 模拟数据集的端元是三维的，而模型只能生成二维的端元数据。
    """

    def compute_SAD(self, E_true, E_pred, prop: HsiPropertyEnum = HsiPropertyEnum.E): ...
    """
        计算SAD, 适用计算的情况：参与计算的数据都是二维，或都是三维的
    """

    def compute_SRE(self, A_true, A_pred): ...
    """
        计算SRE
    """
