import os

# 获取父目录的父目录的绝对地址,即Study2的绝对地址
current_dir = os.path.dirname(os.path.abspath(__file__))
anchor = os.path.abspath(os.path.join(current_dir, "..", ".."))
anchor = anchor.replace('\\', '/')
# 文件/目录的绝对地址
MAIN_CONFIG_FILE = f'{anchor}/config/main_config.yaml'
MAIN_CONFIG_DIR = f'{anchor}/config/'
DATA_DIR = f'{anchor}/data/initData/'
METHODS_CONFIG_DIR = f'{anchor}/config/methods/'
RESULTS_DRAW_CONFIG_FILE = f'{anchor}/config/results/draw.yaml'
RESULTS_METRICS_CONFIG_FILE = f'{anchor}/config/results/metrics.yaml'
RESULTS_INIT_CONFIG_FILE = f'{anchor}/config/prepare/init.yaml'
