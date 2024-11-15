from core import ModeAdapter
from .draw import Draw
from .metrics import Metrics
import os
import scipy.io as sio
from datetime import datetime
import time
from tqdm import tqdm

cp = ModeAdapter()


class RunMode:
    def __init__(self):
        self.draw = Draw  # 直接赋值类名即可，但写法参考Draw
        self.metrics = Metrics  # 直接赋值类名即可，但写法参考Metrics

    @staticmethod
    def __get_current_date_and_time():
        start_time = datetime.now()
        return start_time.strftime("%Y-%m-%d %H:%M:%S")

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
        start_time = self.__get_current_date_and_time()
        cp.set_seed()
        dataset = cp.get_Dataset()
        initData = cp.get_InitData(dataset, replace=False)
        outdir = cp.get_outdir()
        model = cp.get_Model()
        params = cp.get_params()
        cp.record(outdir)
        print('*' * 60 + '  Start traing!  ' + '*' * 60)
        cp.set_seed()
        data_pred = cp.run(model, params, initData, savepath=outdir, output_display=True)
        data_pred = cp.sort_EndmembersAndAbundances(dataset, data_pred)
        # sort_EndmembersAndAbundances(dtrue=dataset, dpred=datapred, edm_repeat=True, case=2)
        sio.savemat(outdir + "results.mat", data_pred.__dict__)
        ed = time.time()
        end_time = self.__get_current_date_and_time()
        total_time = self.__get_runtime(st, ed)
        print('*' * 60 + '  Execution Time!  ' + '*' * 60)
        print(f'起始时间: {start_time}')
        print(f'终止时间: {end_time}')
        print(f'共计时间: {total_time}')
        cp.record_inyaml(outpath=outdir, content=f'start_time: {start_time}')
        cp.record_inyaml(outpath=outdir, content=f'end_time: {end_time}')
        cp.record_inyaml(outpath=outdir, content=f'total_time: {total_time}')
        print('*' * 60 + '  Metrics!  ' + '*' * 60)
        # 值得回顾的办法：采用配置文件方式读取，已废弃
        # cp.compute(dataset, data_pred, outdir)
        # cp.draw(dataset, data_pred, outdir + "/assets")
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
        if cp.cfg.output.draw:
            draw_dir = out_path + '/assets'
            if not os.path.exists(draw_dir):
                os.makedirs(draw_dir)
            d()
