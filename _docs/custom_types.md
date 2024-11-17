# custom_types

## HsiDataset

这个类适用场景有：数据集、初始化数据以及实验结果。
这个类支持以下字段
- Y / pixels
- A / abu
- E / edm
- D
- H / imgHeight
- W / imgWidth
- L / bandNum
- P / edmNum
- N / pixelNum
- name
- other
假设有个实例
```angular2html
hd = HsiDataset(...)
# 以下三种情况会得到同样的结果
hd.A
hd.abu
hd["A"]

```