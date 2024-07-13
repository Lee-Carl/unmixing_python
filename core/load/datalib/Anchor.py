import os

# 作用：根据位置情况，定位数据集的位置
# 获取父目录的父目录的绝对地址,即Study2的绝对地址
current_dir = os.path.dirname(os.path.abspath(__file__))
anchor = os.path.abspath(os.path.join(current_dir, "..", ".."))
anchor = anchor.replace('\\', '/')

DATASET_DIR = anchor
DATALIB_DIR = f'{anchor}/data/datalib/'
