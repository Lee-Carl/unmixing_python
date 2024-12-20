from core import ModeAdapter, DataProcessor, MyLog
from .draw import Draw
from .metrics import Metrics
import os
import scipy.io as sio
from datetime import datetime
import time
from tqdm import tqdm
from utils import HsiUtil, TimeUtil
from custom_types import MainCfg

dp = DataProcessor()


class ParamsMode:
    def __init__(self, cfg: MainCfg):
        self.metrics = Metrics
        self.draw = Draw
        self.cp = ModeAdapter(cfg)
        self.log = MyLog(cfg)

    @staticmethod
    def __get_runtime(st, ed):
        execution_time = ed - st
        hours, remainder = divmod(execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}:{int(minutes)}:{seconds:.2f}"

    def run(self):
        st = time.time()
        start_time = TimeUtil.getCurrentTime()
        self.cp.set_seed()
        dataset = self.cp.get_Dataset()
        initData = self.cp.get_InitData(dataset)
        outdir = self.log.get_outdir()
        model = self.cp.get_Model()
        params = self.cp.get_params()
        obj, around = self.cp.get_Params_adjust()

        if obj not in params.keys():
            raise ValueError(
                f'Cannot find the parameter \'{obj}\' which you could want to adjust, please check the yaml of the method!')

        # 记录参数
        ar, es, yr, ys = [], [], [], []
        iter = []
        lams = []
        self.log.record(outdir)
        print('*' * 60 + '  Start traing!  ' + '*' * 60)
        time.sleep(0.1)
        progress = tqdm(around, desc='Params Loop')

        start_time = datetime.now()  # 获取开始时间
        for i, e in enumerate(progress):
            params[obj] = e
            self.cp.set_seed()

            datapred = self.cp.run(model, params, initData, savepath=outdir, output_display=False)
            datapred = HsiUtil.checkHsiDatasetDims(datapred)
            datapred = self.cp.sort_EndmembersAndAbundances(dataset, datapred)

            mt = self.metrics(dataset, datapred)
            d = mt()
            armse_a, asad_e, armse_y, asad_y = d['armse_a'], d['asad_e'], d['armse_y'], d['asad_y']

            # 显示某一具体时间完成
            elapsed_time = datetime.now() - start_time  # 跑到第i轮所用的总时长
            estimated_time_remaining = elapsed_time * (len(around) / (i + 1)) - elapsed_time
            estimated_completion_time = datetime.now() + estimated_time_remaining

            # 设置描述信息以显示预计完成时间
            progress.set_description(
                f"lam={e:.3e}, Estimated completion time: {estimated_completion_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )

            lams.append(e)
            ar.append(armse_a)
            es.append(asad_e)
            yr.append(armse_y)
            ys.append(asad_y)
            iter.append(i)
        params_dir = os.path.join(outdir, "results.mat")
        sio.savemat(params_dir, {
            "lam": lams,
            "aRMSE_A": ar,
            "aSAD_E": es,
            "aRMSE_Y": yr,
            "aSAD_Y": ys,
            "iter": iter
        })
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
        print('*' * 60 + '  Analysis!  ' + '*' * 60)
        self.analysis_params(outdir)

    def analysis_params(self, outdir):
        p = self.draw(outdir)
        p()
