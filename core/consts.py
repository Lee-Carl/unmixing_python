import os

# 作用：根据位置情况，定位数据集的位置
# 获取父目录的父目录的绝对地址,即绝对地址
current_dir = os.path.dirname(os.path.abspath(__file__))
anchor = os.path.abspath(os.path.join(current_dir, ".."))
anchor = anchor.replace('\\', '/')

DATASET_DIR = anchor
MAIN_CONFIG_DIR = f'{anchor}/config/'
INITDATA_DIR = f'{anchor}/data/initData/'
DATALIB_DIR = f'{anchor}/data/datalib/'
METHODS_CONFIG_DIR = f'{anchor}/config/methods/'
RESULTS_DRAW_CONFIG_FILE = f'{anchor}/config/results/draw.yaml'
RESULTS_METRICS_CONFIG_FILE = f'{anchor}/config/results/metrics.yaml'
RESULTS_INIT_CONFIG_FILE = f'{anchor}/config/prepare/init.yaml'
REAL_DATASET_DIR = f'{anchor}/data/dataset/real/'
SIMULATED_DATASET_DIR = f'{anchor}/data/dataset/simulate/'
DATA_LOADER_CONFIG = f'{anchor}/config/setting/dataset_loader.yaml'
RESULTS_DIR = f'{anchor}/res/'
