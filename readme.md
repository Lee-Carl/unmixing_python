# 解混_Python

![](https://img.shields.io/badge/language-python-brightgreen)
![Python Version](https://img.shields.io/badge/Python-%E2%89%A5%203.7-blue.svg?logo=python)
![](https://img.shields.io/badge/PyTorch-%E2%89%A5%201.0.0-red.svg?logo=pytorch)
![NumPy](https://img.shields.io/badge/NumPy-%E2%89%A5%201.21.5-orange.svg?logo=numpy)

[中文](readme.md) | [English](readme.en.md)

## 1.简介

之前做科研的时候，我非常头疼切换不同的方法、数据集、损失函数时，得不断地添加或调整代码。在接触大量的代码后，我学到了一些写解混代码的理念以及产生了一个idea, 那就是：写一个代码框架，每增加一个数据集，指标等，都可以用在所有需要进行的解混实验中；每添加一个解混方法，都可以封装好的的数据集、指标等。这样做，可以让我最小程度地思考如何调整代码，而是最大程度地思考如何调整模型或解混方法。此外，这里还分享了一些我认为特别有用的github库。

**简单来说**，这是一个与高光谱图像解混技术相关的库。这个库有两个作用：
- 提供一个进行解混实验的代码方案
- 分享一些非常有价值的库

## 2.代码

### 2.1 安装及使用

- 通过requirements.txt安装py库，或者自行安装（建议版本至少为python3.7+pytorch 1.0.0）；
- 在`main.py`中写配置信息，然后点击运行即可
- [说明文档](_docs/unmixing_python_api.pyi)

### 2.2 代码方案

这个库具有以下的特点：
- 对数据集、解混方法、指标、损失函数等进行了规范，调用方便；
- 切换数据集、解混方法等内容时，只需要更改`main.py`中的信息即可；
- 为解混方法保证了相同的实验环境；
- 以`numpy`数据为主要的数据类型。只有在训练模型时，才会将`numpy`数据转换成`pytorch`数据。

它目前主要的不足是：
- 缺少与光谱库相关的代码；
- 没有考虑处理多个数据源；
- 调参方式比较单一，每次只可以调一个参数。

### 2.3 数据集

以下是可提供的数据集, 放在了百度网盘(提取码: dcn5)：
```
https://pan.baidu.com/s/113ZNvTTxBLb6tZLqAc9Crw?pwd=dcn5
```

此外，可以自行添加数据集。

## 3.值得学习的github库
- Hyperspectral-Imaging ([很全面的库](https://github.com/xianchaoxiu/Hyperspectral-Imaging))

### 3.1 解混
- HySUPP ([非常好的代码库](https://github.com/BehnoodRasti/HySUPP))
- awesome-hyperspectral-image-unmixing
  ([棒](https://github.com/xiuheng-wang/awesome-hyperspectral-image-unmixing))
- PGMSU ([点击这里](https://github.com/shuaikaishi/PGMSU))
- EGU-Net ([点击这里](https://github.com/danfenghong/IEEE_TNNLS_EGU-Net))

### 3.2 分类
- HSI_Classification_Models ([酷](https://github.com/Candy-CY/Hyperspectral-Image-Classification-Models))