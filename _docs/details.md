# 1 配置表

大致的配置表字段如下

```
unmxingInfo = {
    # 数据集
    "dataset": DatasetsEnum.Urban4,
    # 方法
    "method": MethodsEnum.PGMSU,
    # 模式
    "mode": ModeEnum.Run,
    # 随机种子
    "seed": 0,
    # 初始化方式
    "init": {
        "custom_init_data": None,
        "custom_init_method": None,
        "snr": 0,
        "normalization": True,
        "E": InitE_Enum.VCA,
        "A": InitA_Enum.SUnSAL,
        "D": None,
        "show_initdata": True,
    },
    # 输出
    "output": {
        "sort": True,
        "draw": 'default',
        "metrics": 'default',
    },
    # 调参模式的配置
    "params": {
        "obj": "lambda_kl",
        "around": around
    }
}
```

## 1.1 填写

配置表字段写在了py文件中，之所以这样做，而不用yaml文件填写，是因为对于需要频繁更改的字段，我希望通过枚举变量完成，这样做可以让编译器告诉我是否填写正确。
换句话说，对于配置表中的`dataset,method,mode`以及`init`字段下的`A,E`字段都可以用枚举变量进行填写。这些枚举变量放在了`custom_types/enums.py`文件中，可根据具体的需求增加方法、数据集等等。

然后是各个字段的含义：

- dataset (IntEnum) - 数据集，这里填写对应的枚举变量
- method (IntEnum) - 解混方法，这里填写对应的枚举变量
- mode (IntEnum) - 模式，主要写的是Run和Param（调参）模式，这里填写对应的枚举变量
- seed (int) - 随机种子，用于固定结果。有些死板的是，在得到初始化结果、训练模型前都会使用这里的seed作为参数固定随机种子。
- init - 初始化配置字段
    - custom_init_data (str): 存放在data/initdata下的初始化数据的路径。用于存放在其他地方生成的初始化数据，但需要遵循这里的数据集的标签规定。 
    - custom_init_method (str): 未实现功能。用于自定义一个初始化方法
    - snr (float) - 噪声(db)
    - normalization (bool) - 是否对Y归一化
    - E (IntEnum) - 初始化endmembers的方法，这里填写对应的枚举变量
    - A (IntEnum) - 初始化abundances的方法，这里填写对应的枚举变量
    - D - 功能未完成。忽略
    - show_initdata (bool) - 是否显示初始化结果；若为True，则显示，False反之
- output - 控制输入的字段
    - sort (bool) - 是否对输出结果排序，True为排序
    - draw (str) - 未实现功能。
    - metrics (str) - 未实现功能。
- params - 调参模式的字段
    - obj (str) - 参数名称。这个字段与对应方法的参数相关，可以config/methods目录下查找
    - around (Any) - 调参范围，一般是一个list，例如可以写[1e-3,1e-2,1e-1]

# 2 数据集

用于导出数据集, 每个数据集按同一格式存放, 并且有以下要求:

- P: 端元数
- L: 波段数
- N: 图像像素点数
- H: 图像高度
- W: 图像宽度
- Y: 像元数据, 遵循 L * N
- E: 端元数据, 遵循 L * P
- A: 丰度数据, 遵循 P * N
- D: 本意是用于光谱库，但并未实现功能

## 2.1 添加数据集

假设添加的数据集名称为name，对应的导出函数为loader

1. 在data/dataset目录下，添加数据集
2. 在 core/load/real 或 core/load/simu 下写一个导出函数loader，写法参考其他文件;
3. 在config/setting/dataset_loader.yaml中, 添加"name: loader";
4. 在custom_types/enums.py文件下，找到DatasetEnum添加`name = auto()`；至此，添加完成。

# 3 解混方法

## 3.1 添加解混方法

假设解混方法名称为`name`

1. 在methods目录下写解混方法。解混方法都需要继承`MethodBase`类，并实现run方法；
2. 在config/methods目录写一个yaml文件，名称参考`name.yaml`；
3. 在custom_types/enums.py文件中找到MethodsEnum，添加这个解混方法，内容参考`name = auto()`。至此，添加完成。

## 3.2 添加参数

参数文件写在config/methods目录下。需要写一个以解混方法名称命名的yaml文件。这个文件需要写两个字段：
- src - 解混方法的路径
- params - 必须填写一个`default`字段，意味着它可以跑任何数据集，或者认为这是鲁棒性最好的一组参数。如果需要针对不同的数据集使用不同的参数，那么在`src`字段下写对应的数据集名称，然后是它的一组参数