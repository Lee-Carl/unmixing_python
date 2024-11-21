from .load import ModuleLoader
from custom_types import MainCfg, HsiDataset, InitE_Enum, InitA_Enum
from core import consts
from utils import FileUtil, HsiUtil
from .DataProcessor import DataProcessor
import os
import copy

dp = DataProcessor()
ml = ModuleLoader()


class ModeAdapter:
    def __init__(self, cfg: MainCfg):
        self.cfg = cfg

    def set_seed(self, seed: int = 0):
        if seed:
            dp.set_seed(seed)
        else:
            dp.set_seed(self.cfg.seed)

    def __getInitStr(self) -> str:
        # 导出初始化数据名称
        # 如果采用的是自定义数据集（需要满足字段，见data_.yaml），那么直接返回相应初始化方式构成的字段，否则采用main_config.yaml中的初始化方式
        return self.cfg.init.custom_init_data if self.cfg.init.custom_init_data else f'{str(self.cfg.init.snr)}db_{self.cfg.init.E.name}_{self.cfg.init.A.name}'

    def get_Dataset(self) -> HsiDataset:
        return dp.loadDatast(self.cfg.dataset)

    def get_InitData(self, dataset: HsiDataset, replace=False) -> HsiDataset:
        # 数据集名称
        case = self.cfg.dataset
        # 初始化数据信息
        custom_init_data = self.cfg.init.custom_init_data
        custom_init_method = self.cfg.init.custom_init_method
        snr = self.cfg.init.snr
        initE = self.cfg.init.E
        initA = self.cfg.init.A
        normalization = self.cfg.init.normalization

        "---------------------------------------------------------------"
        # 通过初始化配置生成字符串，并通过此字符串导入对应数据
        init_str = custom_init_data if custom_init_data else f'{str(snr)}db_{initE}_{initA}'

        exist_flag, init = dp.getInitData(case.name, init_str)
        # 优先级第一：指定的数据
        if (not exist_flag) and custom_init_data:
            # 不存在，但指定了此数据，则报错
            raise ValueError('Cannot find the init data!')

        # 优先级第二：指定的方法
        elif custom_init_method:
            custom_init_methods_class = ml.get_Init_Function(custom_init_method)
            obj = custom_init_methods_class(dataset)
            init = obj()

        # 优先级第三：指定的初始化
        elif (not exist_flag) or replace:
            # 不存在，或要被替换，生成数据
            print("初始化数据: 正在生成初始化数据...")
            savepos = f'{consts.INITDATA_DIR}/{case.name}/'
            FileUtil.createdir(savepos)
            edmEnum: InitE_Enum = InitE_Enum(initE)
            abuEnum: InitA_Enum = InitA_Enum(initA)
            init = dp.gen_initData(dataset, initE=edmEnum, initA=abuEnum, snr=snr,
                                   normalization=normalization,
                                   seed=self.cfg.seed)
            FileUtil.savemat(f'{savepos}/{init.name}.mat', init.__dict__)
        # if self.cfg.init.normalization:
        #     for key in ['A', 'E', 'Y']:
        #         if key in dataset.keys():
        #             dataset[key] = ip.normalization(dataset[key])
        return init

    def get_Model(self):
        if self.cfg.method:
            return ml.get_Method(method_name=self.cfg.method.name)
        else:
            raise ValueError(f"没有填写{self.cfg.method.name}字段")

    def get_params(self) -> dict:
        datasetName: str = self.cfg.dataset.name
        methodName: str = self.cfg.method.name
        dic: dict = ml.get_Method_params(dataset_name=datasetName, method_name=methodName)
        return dic

    @staticmethod
    def run(method, params: dict, init: HsiDataset, savepath=None, output_display=True) -> HsiDataset:
        model = method(params=params, init=copy.deepcopy(init))
        data_pred: dict = model.run(savepath=savepath, output_display=output_display)
        dic: dict = init.__dict__.copy() # tod
        dic.update(data_pred)
        data = HsiDataset(**dic)
        return HsiUtil.checkHsiDatasetDims(data)

    def get_Params_adjust(self):
        around = self.cfg.params.around
        obj = self.cfg.params.obj
        return obj, around

    @staticmethod
    def sort_EndmembersAndAbundances(dataset: HsiDataset, datapred: HsiDataset) -> HsiDataset:
        datapred = dp.sort_edm_and_abu(dtrue=dataset, dpred=datapred, case=2, edm_repeat=True, abu_repeat=True)
        return datapred

    def compute(self, dataset, datapred, out_path=None):
        compute_way = self.cfg.output.metrics  # 获取配置文件上的指标计算方式
        if compute_way:
            compute_class = ml.get_Metrics_Function(compute_way)  # 获取计算方法的类
            compute_obj = compute_class(dataset, datapred)  # 定义计算方法的对象
            results = compute_obj.__str__()  # 得到字符串形式的结果
            print(results)  # 不用删除，将结果打印到控制台
            FileUtil.writeFile(os.path.join(out_path, 'log.txt'), results)
            return results

    def draw(self, dataset, datapred, out_path=None):
        draw_way: str = self.cfg.output.draw  # 获取配置文件上的指标计算方式
        if draw_way:
            try:
                draw_dir = out_path + '/assets'
                FileUtil.createdir(draw_dir)
                draw_class = ml.get_Draw_Function(draw_way)  # 获取计算方式的类
                draw_obj = draw_class(dataset, datapred, out_path)  # 定义对象
                draw_obj()  # 获取计算结果
            except Exception as e:
                print(f"A question ({e}) occured when drawing!")
