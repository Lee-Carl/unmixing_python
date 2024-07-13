import shutil
# from .processor import InitProcessor, DataProcessor
# from .config import MainConfig, MethodsConfig,
from . import InitProcessor, DataProcessor
from ..config import MainConfig, MethodsConfig, PrepareConfig
import os
import yaml
from typing import List
import re
import copy


class MainConfig_Init:
    def __init__(self, **kwargs):
        self.show_initdata: str = kwargs.get('show_initdata', False)
        self.custom_init_data: str = kwargs.get('custom_init_data', None)
        self.custom_init_method: str = kwargs.get('custom_init_method', None)
        self.snr: float = kwargs.get('snr', 0)
        self.normalization: bool = kwargs.get('normalization', False)
        self.A: str = kwargs.get('A', None)
        self.E: str = kwargs.get('E', None)
        self.D: str = kwargs.get('D', None)


class MainConfig_Output:
    def __init__(self, **kwargs):
        self.draw: bool = kwargs.get('draw', False)
        self.normalization: bool = kwargs.get('normalization', False)
        self.sort: bool = kwargs.get('sort', False)
        self.metrics: str = kwargs.get('metrics', None)


class MainConfig_Params:
    def __init__(self, **kwargs):
        self.obj: str = kwargs.get('obj', None)
        self.around: List[float] = eval(kwargs[kwargs['around']]) if kwargs['around'] else None


class _MainConfig:
    def __init__(self, **kwargs):
        yp = MethodsConfig()
        self.dataset: str = kwargs.get('dataset', None)
        self.init: MainConfig_Init = MainConfig_Init(**kwargs['init']) if kwargs['init'] else None
        self.method = yp.get_Method(method_name=kwargs['method']) if kwargs['method'] else None
        self.mode: str = kwargs.get('mode', None)
        self.params: MainConfig_Params = MainConfig_Params(**kwargs['params']) if kwargs['params'] else None
        self.output: MainConfig_Output = MainConfig_Output(**kwargs['output']) if kwargs['output'] else None
        self.seed: int = kwargs.get('seed', 0)


ip = InitProcessor()  # 初始化信息
mc = MainConfig()  # 获取主配置信息
mec = MethodsConfig()  # 获取方法
res_cfg = PrepareConfig()  # 获取指标的计算方式和画图的方式


class CoreProcessor:
    def __init__(self):
        self.initData_dir = self.__get_abs_pos()  # 初始化数据目录名称

        # 载入配置，为字典类型
        self.cfg_dict = mc.get()

        # 载入配置，为MainConfig
        self.cfg = _MainConfig(**self.cfg_dict)

    @staticmethod
    def __get_abs_pos():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        anchor = os.path.abspath(os.path.join(current_dir, "..", "..", "data", "initData"))
        anchor = anchor.replace('\\', '/')
        return anchor

    def set_seed(self):
        ip.set_seed(self.cfg.seed)

    def __getInitStr(self):
        # 导出初始化数据名称
        # 如果采用的是自定义数据集（需要满足字段，见data_.yaml），那么直接返回相应初始化方式构成的字段，否则采用main_config.yaml中的初始化方式
        return self.cfg.init.custom_init_data if self.cfg.init.custom_init_data else f'{str(self.cfg.init.snr)}db_{self.cfg.init.E}_{self.cfg.init.A}'

    @staticmethod
    def createdir(dn):
        if not os.path.exists(dn):
            os.makedirs(dn)

    def get_Dataset(self):
        return ip.loadhsi(self.cfg.dataset)

    def get_InitData(self, dataset, replace=False):
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

        exist_flag, init = mc.get_InitData(case, init_str)
        # 优先级第一：指定的数据
        if (not exist_flag) and custom_init_data:
            # 不存在，但指定了此数据，则报错
            raise ValueError('Cannot find the init data!')

        # 优先级第二：指定的方法
        elif custom_init_method:
            custom_init_methods_class = res_cfg.get_Init_Function(custom_init_method)
            obj = custom_init_methods_class(dataset)
            init = obj()

        # 优先级第三：指定的初始化
        elif (not exist_flag) or replace:
            # 不存在，或要被替换，生成数据
            print("初始化数据: 正在生成初始化数据...")
            savepos = f'{self.initData_dir}/{case}/'
            self.createdir(savepos)
            init = ip.generateInitData(dataset, initE=initE, initA=initA, snr=snr,
                                       normalization=normalization,
                                       savepath=savepos, seed=self.cfg.seed)
        # if self.cfg.init.normalization:
        #     for key in ['A', 'E', 'Y']:
        #         if key in dataset.keys():
        #             dataset[key] = ip.normalization(dataset[key])
        return init

    def get_Model(self):
        return self.cfg.method

    def get_params(self):
        return mec.get_Method_params(dataset_name=self.cfg.dataset,
                                     method_name=self.cfg.method.__name__)

    def get_outdir(self):
        method = self.cfg.method
        case = self.cfg.dataset
        mode = self.cfg.mode

        # 默认存放的目录
        main_dir = f'res/{method.__name__}/{case}/'

        # 谨防空目录
        self.createdir(main_dir)

        existing_dirs = []
        if mode == 'run':
            for name in os.listdir(main_dir):
                # 1. 是否是目录
                # 2. 是否是数字目录
                if os.path.isdir(main_dir + name) and name.isdigit():
                    existing_dirs.append(name)

            # 确定最大的数字命名
            max_dir_num = max(existing_dirs, default=0, key=int)

            # 新的目录名称
            new_dir_num = int(max_dir_num) + 1

            # 创建新的目录
            out_path = str(new_dir_num)
        else:
            pattern = re.compile(r'params_(\d+)')
            for name in os.listdir(main_dir):
                if os.path.isdir(main_dir + name) and pattern.match(name):
                    name = pattern.search(name).group(1)
                    existing_dirs.append(name)

            # 确定最大的数字命名
            max_dir_num = max(existing_dirs, default=0, key=int)

            # 新的目录名称
            new_dir_num = int(max_dir_num) + 1

            # 创建新的目录
            out_path = 'params_' + str(new_dir_num)

        # 生成最终目录
        out_path = main_dir + out_path + '/'

        # 创建目录
        self.createdir(out_path)
        self.createdir(out_path + "/assets")  # 存放图片的目录

        return out_path

    @staticmethod
    def run(method, params, init, savepath=None, output_display=True):
        model = method(params=params, init=copy.deepcopy(init))
        data_pred = model.run(savepath=savepath, output_display=output_display)
        return data_pred

    def get_Params_adjust(self):
        around = self.cfg.params.around
        obj = self.cfg.params.obj
        return obj, around

    @staticmethod
    def sort_EndmembersAndAbundances(dataset, datapred):
        dp = DataProcessor(dataset)
        if 'E' in datapred and 'A' in datapred:
            pass
            # datapred = dp.sort_EndmembersAndAbundances(dtrue=dataset, dpred=datapred)
            # datapred = dp.sort_EndmembersAndAbundances(dtrue=dataset, dpred=datapred, repeat=False, case=2)
            # datapred = dp.sort_EndmembersAndAbundances(dtrue=dataset, dpred=datapred, repeat=True, case=2)
            datapred = dp.sort_EndmembersAndAbundances(dtrue=dataset, dpred=datapred, edm_repeat=True, case=2)
        return datapred

    def compute(self, dataset, datapred, out_path=None):
        compute_way = self.cfg.output.metrics  # 获取配置文件上的指标计算方式
        if compute_way:
            compute_class = res_cfg.get_Metrics_Function(compute_way)  # 获取计算方法的类
            compute_obj = compute_class(dataset, datapred)  # 定义计算方法的对象
            results = compute_obj.__str__()  # 得到字符串形式的结果
            print(results)  # 不用删除，将结果打印到控制台
            with open(os.path.join(out_path, 'log.txt'), "w") as file:
                file.write(results)
                # file.write(f"Total time taken:{time_string}\n")
            return results

    def draw(self, dataset, datapred, out_path=None):
        draw_way = self.cfg.output.draw  # 获取配置文件上的指标计算方式
        if self.cfg.output.draw:
            draw_dir = out_path + '/assets'
            self.createdir(draw_dir)
            try:
                draw_class = res_cfg.get_Draw_Function(draw_way)  # 获取计算方式的类
                draw_obj = draw_class(dataset, datapred, out_path)  # 定义对象
                draw_obj()  # 获取计算结果
            except Exception as e:
                print(f"A question ({e}) occured when drawing!")

    def record(self, out_path):
        def dict_recursive(dictionary, indent=0):
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    info = ' ' * indent + f'{key}:'
                    print(info)
                    dict_recursive(value, indent + 2)
                else:
                    info = ' ' * indent + f'{key}: {value}'
                    print(info)

        params = self.get_params()
        cfg_dict = mc.get()

        if cfg_dict["mode"] == 'run':
            cfg_dict.pop('params', None)
        else:
            cfg_params = cfg_dict['params']
            cfg_params['around'] = cfg_params[cfg_params['around']]
            for key in ['around1', 'around2', 'around3']:
                cfg_params.pop(key, None)
        cfg_dict['relative_path'] = out_path

        print('*' * 60 + '  Initial Information  ' + '*' * 60)
        dict_recursive(cfg_dict)
        dict_recursive({'params': params})
        # save
        cfg_dict['abs_path'] = os.path.abspath(out_path)
        yaml_dir = os.path.join(out_path, 'config')
        self.createdir(yaml_dir)
        with open(os.path.join(yaml_dir, 'detail.yaml'), "a") as file:
            yaml.dump(cfg_dict, file, default_flow_style=False)
        with open(os.path.join(yaml_dir, 'params.yaml'), "a") as file:
            yaml.dump(params, file, default_flow_style=False)

        if os.path.exists(f'methods/{self.cfg.method.__name__}'):
            shutil.copytree(f'./methods/{self.cfg.method.__name__}', out_path + '/model')

    @staticmethod
    def record_inyaml(content, outpath):
        yaml_dir = os.path.join(outpath, 'config')
        with open(os.path.join(yaml_dir, 'detail.yaml'), "a") as file:
            file.write(f"{content}\n")
