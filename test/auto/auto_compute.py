import os
from core.mode import Auto

if __name__ == '__main__':
    """
    计算全部实验的指标：computed_all(cases,models)
    计算一个实验的指标：computed(case,model)
    画一个模型的图：plots(case, model)
    画多个模型的图：plots_all(cases, models)
    """
    # 0: "Samson",
    # 1: "JasperRidge",
    # 2: "DeepGUn_ex2",
    # 3: "DeepGUn_ex4",
    # 4: "SUnCNN_DC1",
    # 5: "SUnCNN_DC2",
    # 6: "Urban4",
    # 7: "Urban5",
    # 8: "Urban6"

    ex1 = {
        # "datasets": ["DeepGUn_ex2", "JasperRidge"],
        "datasets": ["JasperRidge"],
        "methods": ["ELMM", "GLMM", "PLMM", "MESMA", "DeepGUn", "PGMSU", "DGASU"],
        "edm_name": {
            "JasperRidge": ["Tree", "Water", "Soil", "Road"],
            "DeepGUn_ex2": ["EM #1", "EM #2", "EM #3"]
        }
    }

    ex2 = {
        "datasets": ["DeepGUn_ex4", "Samson"],
        # "datasets": ["DeepGUn_ex4"],
        "methods": ["FCLSU", "ALMM", "CNNAEU", "DAEU", "EndNet", "CyCU", "DCAE"],
        "edm_name": {
            "Samson": ["Soil", "Tree", "Water"],
            "DeepGUn_ex4": ["EM #1", "EM #2", "EM #3", "EM #4", "EM #5"]
        }
    }

    # 设置
    params = {
        'obj_file': 'results.mat',  # 解析的文件，需要包含统一的丰度名称、端元名称
        'src': '../res',
        'dst': '../../_exp/ex2',
        'draw': '_abus1',  # 相对于dst目录下
        'xlsx': False,  # True:将数据打印成xlsx文件
    }

    acp = Auto(params)  # 初始化

    # if not os.path.exists(os.path.join(os.getcwd(), acp.dst)):
    #     # 如果dst目录不存在，就收集最新的目录信息
    #     acp.getLatestDirInfo()

    # 排序
    # acp.sort_all(ex2)

    # 计算指标
    # acp.computed("DeepGUn_ex4", "FCLSU")  # 单一
    # acp.computed_all(ex2)  # 全部

    # 画图
    # acp.plots_one(ex=ex2, types=["abu"], show=False)  # 将一个数据集的所有图分开画
    # acp.plots_all(ex=ex2, types=["abu"], show=False)  # 将一个数据集的所有图画在一个图上
    acp.plots_onepic_abu(ex=ex2, show=True, todiff=True, t=False)  # 将同一数据集下的不同方法画在同一张图上
    # acp.plots_onepic_edm(ex=ex1, show=True)  # 将同一数据集下的不同方法画在同一张图上
