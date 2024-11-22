# conda命令

## 版本

获取版本号

```
conda -V
```

```
conda --version
```


## 环境篇

创建虚拟环境

```
conda create -n your_env_name python=x.x
```

退出虚拟环境

```
conda deactivate
```

查看所有环境

```
conda env list
```

```
conda info --envs
```

切换环境

```
conda activate 名称
```

```
conda config --set auto_activate_base false
```

删除环境

```
conda remove -n 环境名 --all
```

查看某个指定环境的已安装包

```
conda list -n py35
```

### 复制环境

**方法一**

指定目录下生成环境yml文件

```
conda env export > 目录/environment.yml 
```


从yml文件创建环境 

```
conda env create -n env_name -f environment.yml
```

**方法二**

复制本地的环境。这样的好处是可以避免有些包因版本更替而无法从默认地址上下载。
```
conda create --name <新环境名称> --clone <原环境名称>
```


## 镜像源篇

添加镜像源

```
conda config --add channels 网址
```

删除指定镜像

```
conda config --remove channels 网址
```

删除所有镜像

```
conda config --remove-key channels
```

查看所有镜像

```
conda config --show channels
```

常用的镜像（可直接使用）

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
```

其他参考（不可直接使用）
```
channels:
  - defaults
show_channel_urls: true
default_channels:
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

临时更换镜像 ，加入`-c`参数

```
conda install 包名 -c 网址
```


设置搜索是显示通道地址

```
conda config --set show_channel_urls yes
```

## 管理篇

查看安装了哪些包

```text
conda list
```

查看conda源中包的信息

```
conda search package_name
```

安装包

```
conda install 包名
```

卸载包

```
conda uninstall 包名
```

清理无用的安装包

```
conda clean -p
```


清理tar包

```
conda clean -t
```


清理所有安装包及cache

```
conda clean -y --all
```


更新conda，保持conda最新

```
conda update conda
```
