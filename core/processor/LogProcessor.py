import datetime
import time
import os
import shutil


class LogProcessor:
    def __init__(self):
        self.begin_time = None
        self.end_time = None

    def tic(self):
        self.begin_time = time.time()

    def toc(self):
        pass

    @staticmethod
    def createdir(dn):
        if not os.path.exists(dn):
            os.makedirs(dn)

    @staticmethod
    def getTime(time_format="%Y-%m-%d %H:%M:%S") -> str:
        return datetime.datetime.now().strftime(time_format)

    @staticmethod
    def getTime_str(time) -> str:
        return str(datetime.timedelta(seconds=time))

    @staticmethod
    def copy_dir(src, dst):
        # 获取源目录下的所有文件
        shutil.copytree(src, dst)
