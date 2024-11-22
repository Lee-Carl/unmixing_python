from core import ModeAdapter, MyLog
from custom_types import MainCfg
from .draw import Draw
from .metrics import Metrics
import os
import scipy.io as sio
from utils import TimeUtil
import time


class RunMode:
    def __init__(self, cfg: MainCfg):
        self.draw = Draw  # 直接赋值类名即可，但写法参考Draw
        self.metrics = Metrics  # 直接赋值类名即可，但写法参考Metrics
        self.cp = ModeAdapter(cfg)
        self.log = MyLog(cfg)

    @staticmethod
    def __get_runtime(st, ed):
        execution_time = ed - st
        hours, remainder = divmod(execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}:{int(minutes)}:{seconds:.2f}"

    @staticmethod
    def test_initData(dataset, initdata, skip=False):
        dm = Metrics(dataset, initdata)
        results = dm.__str__()
        print(results)
        dd = Draw(dataset, initdata)
        dd()
        if skip:
            input("Press any key to continue ...")

    def run(self):
        st = time.time()
        start_time = TimeUtil.getCurrentTime()
        self.cp.set_seed()
        dataset = self.cp.get_Dataset()
        initData = self.cp.get_InitData(dataset, replace=False)
        outdir = self.log.get_outdir()
        model = self.cp.get_Model()
        params = self.cp.get_params()
        self.log.record(outdir)
        print('*' * 60 + '  Start traing!  ' + '*' * 60)
        self.cp.set_seed()
        data_pred = self.cp.run(model, params, initData, savepath=outdir, output_display=True)
        data_pred = self.cp.sort_EndmembersAndAbundances(dataset, data_pred)
        # sort_EndmembersAndAbundances(dtrue=dataset, dpred=datapred, edm_repeat=True, case=2)
        sio.savemat(outdir + "results.mat", data_pred.__dict__)
        ed = time.time()
        end_time = TimeUtil.getCurrentTime()
        total_time = self.__get_runtime(st, ed)
        print('*' * 60 + '  Execution Time!  ' + '*' * 60)
        print(f'起始时间: {start_time}')
        print(f'终止时间: {end_time}')
        print(f'共计时间: {total_time}')
        self.log.record_inyaml(outpath=outdir, content=f'start_time: {start_time}')
        self.log.record_inyaml(outpath=outdir, content=f'end_time: {end_time}')
        self.log.record_inyaml(outpath=outdir, content=f'total_time: {total_time}')
        print('*' * 60 + '  Metrics!  ' + '*' * 60)
        # 值得回顾的办法：采用配置文件方式读取，已废弃
        # self.cp.compute(dataset, data_pred, outdir)
        # self.cp.draw(dataset, data_pred, outdir + "/assets")
        # 现在的办法：直接在类中定义
        self.get_Metrics(dataset, data_pred, outdir)
        self.get_Pictures(dataset, data_pred, outdir + "/assets")

    def get_Metrics(self, dataset, datapred, out_path=None):
        m = self.metrics(dataset, datapred)
        results = m.__str__()  # 得到字符串形式的结果
        print(results)  # 不用删除，将结果打印到控制台
        with open(os.path.join(out_path, 'log.txt'), "w") as file:
            file.write(results)
            # file.write(f"Total time taken:{time_string}\n")
        return results

    def get_Pictures(self, dataset, datapred, out_path=None):
        d = self.draw(dataset, datapred, out_path + "/assets")
        if self.cp.cfg.output.draw:
            draw_dir = out_path + '/assets'
            if not os.path.exists(draw_dir):
                os.makedirs(draw_dir)
            d()
