# pip命令

## 版本篇

查看pip版本

```python3
pip --version
```

## 管理篇


列出已安装的包

```
pip list
```

安装包

```
pip install 包名
```

```
pip i 包名
```


卸载包

```
pip uninstall 包名
```


升级pip

```
pip install -U pip
```


临时更换源

```pip
pip install 库包名 -i 网址
```

常用命令（可直接使用）
```pip
pip install 库包名 -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
```

### requirements方式

将安装包记载到requirements.txt文件后，利用pip直接读取文件进行相应地下载

**命令**
```
pip install -r requirements.txt
```

*参数*
- -r, --requirement：
	- Install from the given requirements file. This option can be used multiple times.  
- -t：
	- 安装到指定位置
- -e：
	- 安装可编辑的包。不同项目，但是一个项目依赖时使用

## 镜像源

豆瓣
```
https://pypi.douban.com/simple/
```

清华大学 
```
https://pypi.tuna.tsinghua.edu.cn/simple/
```

阿里云 
```
https://mirrors.aliyun.com/pypi/simple/
```

中国科技大学 
```
https://pypi.mirrors.ustc.edu.cn/simple/
```


中国科技大学 
```
https://pypi.mirrors.ustc.edu.cn/simple/
```

华中理工大学
```
http://pypi.hustunique.com/
```

山东理工大学
```
http://pypi.sdutlinux.org/
```

