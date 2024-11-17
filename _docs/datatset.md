# 数据集

用于导出数据集, 每个数据集按同一格式存放, 并且有以下要求:

- P: 端元数
- L: 波段数
- N: 图像像素点数
- H: 图像高度
- W: 图像宽度
- Y: 像元数据, 遵循 L * N
- E: 端元数据, 遵循 L * P
- A: 丰度数据, 遵循 P * N
- D: 本意是写光谱库，用于稀疏解混，但并未将功能完成

## 添加数据集

假设添加的数据集名称为name，对应的导出函数为loader

1. 在data/dataset目录下，添加数据集
2. 在 core/load/real 或 core/load/simu 下写一个导出函数loader，写法参考其他文件;
3. 在config/setting/dataset_loader.yaml中, 添加"name: loader";
4. 在custom_types/enums.py文件下，找到DatasetEnum,添加`name = auto()`
5. 至此，添加完成
