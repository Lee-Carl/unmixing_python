# Unmixing_Python

![](https://img.shields.io/badge/language-python-brightgreen)
![Python Version](https://img.shields.io/badge/Python-%E2%89%A5%203.7-blue.svg?logo=python)
![](https://img.shields.io/badge/PyTorch-%E2%89%A5%201.0.0-red.svg?logo=pytorch)
![NumPy](https://img.shields.io/badge/NumPy-%E2%89%A5%201.21.5-orange.svg?logo=numpy)

[中文](readme.md) | [English](readme.en.md)

## 1.Introduction
This is a library related to hyperspectral image unmixing techniques. This library serves two purposes:
- Provide a code program for conducting unmixing experiments
- To share some very valuable libraries
<br>

## 2. Code

### 2.1 Installation and usage

- Install the py library via requirements.txt, or install them one by one (recommended version is at least python3.7+pytorch 1.0.0);
- Write the configuration information in `main.py` and just click run
- [Documentation](_docs/unmixing_python_api.pyi)

### 2.2 The code program

This library has the following features:
- Datasets, unmixing methods, metrics, loss functions, etc. are standardized and easy to call;
- When switching datasets, unmixing methods, etc., only the information in `main.py` needs to be changed;
- The same experimental environment is guaranteed for unmixing methods;
- Using `numpy` data as the main data type. Converts `numpy` data to `pytorch` data only when training the model.

Its main shortcoming at the moment is:
- Lack of code related to spectral libraries;
- Doesn't consider handling multiple data sources;
- Calling parameters is rather monolithic, only one parameter can be called at a time.

Translated with DeepL.com (free version)

### 2.3 Datasets

The following datasets are available on a cloud disk (Extract code: dcn5):
``
https://pan.baidu.com/s/113ZNvTTxBLb6tZLqAc9Crw?pwd=dcn5
```

In addition, you can add your own datasets.


## 3.Nice github repositories
- Hyperspectral-Imaging ([comprehensive knowledge repository](https://github.com/xianchaoxiu/Hyperspectral-Imaging))

### 3.1 Unmixing
- HySUPP ([a great code program](https://github.com/BehnoodRasti/HySUPP))
- awesome-hyperspectral-image-unmixing
  ([awesome](https://github.com/xiuheng-wang/awesome-hyperspectral-image-unmixing))
- PGMSU ([click here](https://github.com/shuaikaishi/PGMSU))
- EGU-Net ([click here](https://github.com/danfenghong/IEEE_TNNLS_EGU-Net))

### 3.2 Classification
- HSI_Classification_Models ([cool](https://github.com/Candy-CY/Hyperspectral-Image-Classification-Models))