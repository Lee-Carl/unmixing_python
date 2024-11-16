from core.SmartMetrics import SmartMetrics


class SUnCNN:
    def __init__(self, dtrue, dpred):
        self.dtrue = dtrue
        self.dpred = dpred

    def __str__(self) -> str:
        d = self.__call__()
        results = f'SRE:{d["SRE"]} | {d["RMSE"]}\n'
        return results

    def __call__(self):
        data_true = self.dtrue
        data_pred = self.dpred

        sm = SmartMetrics(data_true)
        A_pred = data_pred["A"]
        A_true = data_true["A"]
        SRE = sm.compute_SRE(A_true, A_pred)
        RMSE = sm.compute_RMSE(A_true, A_pred)

        d = {
            'SRE': SRE,
            'RMSE': RMSE
        }

        return d
