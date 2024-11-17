# loader

用于导出数据集, 每个数据集按同一格式存放, 并且有以下要求:
- P: 端元数
- L: 波段数
- N: 图像像素点数
- H: 图像高度
- W: 图像宽度
- Y: 像元数据, 遵循 L * N
- E: 端元数据, 遵循 L * P
- A: 丰度数据, 遵循 P * N

## 通过loadhsi导出新数据集

1. 在config/setting/dataset_loader.yaml中, 按"数据集名称: 导出函数位置"填写
2. 在core/loader目录下, 在上述拟定的导出函数位置处写导出函数

## loadhsi的api

loadhsi(case: str)
- Args
  - case: 数据集名称, 必须同上述的"config/setting/dataset_loader.yaml"中写的数据集名称一致 
- Return
  - 一个字典