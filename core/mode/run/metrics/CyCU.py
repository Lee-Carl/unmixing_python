from typing import Dict

from core.SmartMetrics import SmartMetrics


class CyCU:
    def __init__(self, dtrue, dpred):
        self.dtrue = dtrue
        self.dpred = dpred

    def __call__(self) -> Dict:
        dtrue = self.dtrue
        dpred = self.dpred

        sm = SmartMetrics(dtrue)
        # 预测数据
        armse_a = 0
        asad_em = 0
        armse_y = 0
        asad_y = 0

        # 计算丰度
        if 'A' in dpred.keys():
            A_pred = dpred["A"]
            A_true = dtrue["A"]
            armse_a = sm.compute_RMSE(A_true, A_pred)

        # 计算端元
        if 'E' in dpred.keys():
            E_pred = dpred["E"]
            if len(E_pred.shape) == 3 and "E_3d" in dtrue.keys():
                E_true = dtrue["E_3d"]
            else:
                E_true = dtrue["E"]
            asad_em = sm.compute_SAD(E_true, E_pred)[0]

        # 计算像元
        if 'Y' in dpred.keys() or ('A' in dpred.keys() and 'E' in dpred.keys()):
            if 'Y' in dpred.keys():
                Y_pred = dpred['Y']
            else:
                Y_pred = sm.dp.generateY(e=dpred["E"], a=dpred["A"])

            armse_y = sm.compute_RMSE_2(dtrue["Y"], Y_pred)
            asad_y = sm.compute_SAD(dtrue["Y"], Y_pred, type="Y")[0]

        d = {
            'armse_a': armse_a,
            'asad_e': asad_em,
            'armse_y': armse_y,
            'asad_y': asad_y
        }
        return d

    def __str__(self) -> str:
        d = self.__call__()
        results = f"Results:\n" \
                  f"\tarmse_a:{d['armse_a']} | asad_em:{d['asad_e']} | sum:{d['armse_a'] + d['asad_e']}\n" \
                  f"\tarmse_y:{d['armse_y']} | asad_y:{d['asad_y']}\n"
        return results
