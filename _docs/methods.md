# 解混方法


## 添加解混方法
假设解混方法名称为`name`
1. 在methods目录下写解混方法。解混方法都需要继承`MethodBase`类，并实现run方法。
2. 在config/methods目录写一个yaml文件，名称参考`name.yaml`
3. 在custom_types/enums.py文件中找到MethodsEnum，添加这个解混方法，内容参考`name = auto()`

这样就完成了添加解混方法