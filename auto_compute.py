import os
from custom_types import HsiPropertyEnum
from mode.auto import AutoMode
from core import draw
from core import SmartMetrics

if __name__ == '__main__':
    ex1 = {
        "datasets": ["JasperRidge"],
        "methods": ["ELMM", "GLMM", "PLMM", "MESMA", "DeepGUn", "PGMSU", "DGASU"],
        "edm_name": {
            "JasperRidge": ["Tree", "Water", "Soil", "Road"],
            "DeepGUn_ex2": ["EM #1", "EM #2", "EM #3"]
        }
    }

    ex2 = {
        "datasets": ["DeepGUn_ex4", "Samson"],
        "methods": ["FCLSU", "ALMM", "CNNAEU", "DAEU", "EndNet", "CyCU", "DCAE"],
        "edm_name": {
            "Samson": ["Soil", "Tree", "Water"],
            "DeepGUn_ex4": ["EM #1", "EM #2", "EM #3", "EM #4", "EM #5"]
        },
        "metrics": [
            {"SAD_Y", "Y", SmartMetrics.compute_SAD},
            {"RMSE_Y", "Y", SmartMetrics.compute_RMSE},
            {"SAD", "E", SmartMetrics.compute_SAD},
            {"A_aRMSE", "A", SmartMetrics.compute_RMSE},
            {"RMSE", "A", SmartMetrics.compute_RMSE},
            {"RMSE2", "A", SmartMetrics.compute_RMSE_2},
        ],
        "draw": [
            draw.abundanceMap()
        ]
    }

    # 设置
    params = {
        'dst': '../../_exp/ex2',
        'draw': '_abus1',  # 相对于dst目录下
        'xlsx': False,  # True:将数据打印成xlsx文件
    }

    # 画图
    # acp.plots_one(ex=ex2, types=["abu"], show=False)  # 将一个数据集的所有图分开画
    # acp.plots_all(ex=ex2, types=["abu"], show=False)  # 将一个数据集的所有图画在一个图上
    # acp.plots_onepic_abu(ex=ex2, show=True, todiff=True, t=False)  # 将同一数据集下的不同方法画在同一张图上
    # acp.plots_onepic_edm(ex=ex1, show=True)  # 将同一数据集下的不同方法画在同一张图上
