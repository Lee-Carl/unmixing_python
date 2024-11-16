import shutil

from .config import MainConfig, MethodsConfig, PrepareConfig
from custom_types import MainCfg, HsiDataset
from utils import FileUtil
from .DataProcessor import DataProcessor
from .load import loadhsi
import os
import re
import copy

mc = MainConfig()  # 获取主配置信息
mec = MethodsConfig()  # 获取方法
res_cfg = PrepareConfig()  # 获取指标的计算方式和画图的方式
dp = DataProcessor()


class ModeAdapter:
    def __init__(self):
        self.initData_dir = FileUtil.getAbsPath_ByRelativepath("../../data/initData")  # 初始化数据目录名称

        # 载入配置，为字典类型
        self.cfg_dict = mc.get()
        self.method = mec.get_Method(method_name=self.cfg_dict['method']) if self.cfg_dict['method'] else None
        # 载入配置，为MainConfig
        self.cfg = MainCfg(**self.cfg_dict)

    def set_seed(self):
        dp.set_seed(self.cfg.seed)

    def __getInitStr(self):
        # 导出初始化数据名称
        # 如果采用的是自定义数据集（需要满足字段，见data_.yaml），那么直接返回相应初始化方式构成的字段，否则采用main_config.yaml中的初始化方式
        return self.cfg.init.custom_init_data if self.cfg.init.custom_init_data else f'{str(self.cfg.init.snr)}db_{self.cfg.init.E}_{self.cfg.init.A}'

    def get_Dataset(self):
        loadhsi(self.cfg.dataset)

    def get_InitData(self, dataset: HsiDataset, replace=False):
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
            FileUtil.createdir(savepos)
            init = dp.gen_initData(dataset, initE=int(initE), initA=int(initA), snr=snr,
                                   normalization=normalization,
                                   seed=self.cfg.seed)
            FileUtil.savemat(f'{savepos}/{init.name}.mat', init.__dict__)
        # if self.cfg.init.normalization:
        #     for key in ['A', 'E', 'Y']:
        #         if key in dataset.keys():
        #             dataset[key] = ip.normalization(dataset[key])
        return init

    def get_Model(self):
        return self.method

    def get_params(self):
        return mec.get_Method_params(dataset_name=self.cfg.dataset,
                                     method_name=self.method.__name__)

    def get_outdir(self):
        method = self.method
        case = self.cfg.dataset
        mode = self.cfg.mode

        # 默认存放的目录
        main_dir = f'res/{method.__name__}/{case}/'

        # 谨防空目录
        FileUtil.createdir(main_dir)

        existing_dirs = []
        if mode == 'run':
            for name in os.listdir(main_dir):
                # 1. 是否是目录
                # 2. 是否是数字目录
                if os.path.isdir(main_dir + name) and name.isdigit():
                    existing_dirs.append(name)

            # 确定最大的数字命名
            max_dir_num = max(existing_dirs, default=0, key=int)

            # 创建新的目录
            out_path = f'{int(max_dir_num) + 1}'
        else:
            pattern = re.compile(r'params_(\d+)')
            for name in os.listdir(main_dir):
                if os.path.isdir(main_dir + name) and pattern.match(name):
                    name = pattern.search(name).group(1)
                    existing_dirs.append(name)

            # 确定最大的数字命名
            max_dir_num = max(existing_dirs, default=0, key=int)

            # 创建新的目录
            out_path = f'params_{int(max_dir_num) + 1}'

        # 生成最终目录
        out_path = main_dir + out_path + '/'

        # 创建目录
        FileUtil.createdir(out_path + "/assets")  # 存放图片的目录

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
    def sort_EndmembersAndAbundances(dataset: HsiDataset, datapred: HsiDataset) -> HsiDataset:
        datapred = dp.sort_edm_and_abu(dtrue=dataset, dpred=datapred, edm_repeat=True, case=2)
        return datapred

    def compute(self, dataset, datapred, out_path=None):
        compute_way = self.cfg.output.metrics  # 获取配置文件上的指标计算方式
        if compute_way:
            compute_class = res_cfg.get_Metrics_Function(compute_way)  # 获取计算方法的类
            compute_obj = compute_class(dataset, datapred)  # 定义计算方法的对象
            results = compute_obj.__str__()  # 得到字符串形式的结果
            print(results)  # 不用删除，将结果打印到控制台
            FileUtil.writeFile(os.path.join(out_path, 'log.txt'), results)
            return results

    def draw(self, dataset, datapred, out_path=None):
        draw_way = self.cfg.output.draw  # 获取配置文件上的指标计算方式
        if self.cfg.output.draw:
            draw_dir = out_path + '/assets'
            FileUtil.createdir(draw_dir)
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
        FileUtil.createdir(yaml_dir)
        FileUtil.writeYamlFile(os.path.join(yaml_dir, 'detail.yaml'), cfg_dict, 'a')
        FileUtil.writeYamlFile(os.path.join(yaml_dir, 'params.yaml'), params, 'a')

        if os.path.exists(f'methods/{self.method.__name__}'):
            shutil.copytree(f'./methods/{self.method.__name__}', out_path + '/model')

    @staticmethod
    def record_inyaml(content, outpath):
        yaml_dir = os.path.join(outpath, 'config')
        with open(os.path.join(yaml_dir, 'detail.yaml'), "a") as file:
            file.write(f"{content}\n")

    @staticmethod
    def copeWithData(data, snr=0, normalization=True):
        data = dp.addNoise(data, snr)
        data = dp.norm(data, normalization)
        return data
