from utils import FileUtil
import os
import re
from custom_types import MainCfg, ModeEnum
import shutil
from .load import ModuleLoader

ml = ModuleLoader()


class MyLog:
    def __init__(self, cfg: MainCfg):
        self.cfg = cfg
        self.outDir = self.__genOutDir()

    def __genOutDir(self) -> str:
        methodName: str = self.cfg.method.name
        datasetName: str = self.cfg.dataset.name
        main_dir: str = f'res/{methodName}/{datasetName}/'
        FileUtil.createdir(main_dir)
        existing_dirs = []
        if self.cfg.mode == ModeEnum.Run:
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

    def get_outdir(self):
        return self.outDir

    @classmethod
    def __dict_recursive(cls, dictionary, indent=0):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                info = ' ' * indent + f'{key}:'
                print(info)
                cls.__dict_recursive(value, indent + 2)
            else:
                info = ' ' * indent + f'{key}: {value}'
                print(info)

    def record(self, out_path):
        datasetName: str = self.cfg.dataset.name
        methodName: str = self.cfg.method.name

        dic: dict = ml.get_Method_params(dataset_name=datasetName, method_name=methodName)

        params = dic
        cfg_dict = self.cfg.__dict__.copy()
        if self.cfg.mode == ModeEnum.Run:
            cfg_dict.pop('params', None)
        else:
            cfg_params = cfg_dict['params']
            cfg_params['around'] = cfg_params[cfg_params['around']]
        cfg_dict['relative_path'] = out_path
        cfg_dict['dataset'] = self.cfg.dataset.name
        cfg_dict['method'] = self.cfg.method.name
        cfg_dict['mode'] = self.cfg.mode.name
        cfg_dict['init'] = self.cfg.init.__dict__

        cfg_dict['init']['A'] = self.cfg.init.A.name
        cfg_dict['init']['E'] = self.cfg.init.E.name
        cfg_dict['output'] = self.cfg.output.__dict__
        print('*' * 60 + '  Initial Information  ' + '*' * 60)
        self.__dict_recursive(cfg_dict)
        self.__dict_recursive({'params': params})
        # save
        cfg_dict['abs_path'] = os.path.abspath(out_path)
        yaml_dir = os.path.join(out_path, 'config')
        FileUtil.createdir(yaml_dir)

        FileUtil.writeYamlFile(os.path.join(yaml_dir, 'detail.yaml'), cfg_dict, 'a')
        FileUtil.writeYamlFile(os.path.join(yaml_dir, 'params.yaml'), params, 'a')
        if os.path.exists(f'methods/{methodName}'):
            shutil.copytree(f'./methods/{methodName}', out_path + '/model')

    @staticmethod
    def record_inyaml(content, outpath):
        yaml_dir = os.path.join(outpath, 'config')
        with open(os.path.join(yaml_dir, 'detail.yaml'), "a") as file:
            file.write(f"{content}\n")
