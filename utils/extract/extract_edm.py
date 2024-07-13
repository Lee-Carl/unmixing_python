import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def extract_edm(y, a):
    """
    y & a 必须是归一化的结果
    input:
        y: true mixed pixels (L,N)
        a: estimated abundances (P,N)
    output:
        E_solution: estimated endmembers (L,P)
    """
    # 检查GPU是否可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据迁移到GPU
    Y = torch.from_numpy(y.copy().astype(np.float32)).to(device)
    A = torch.from_numpy(a.copy().astype(np.float32)).to(device)

    # 使用 Xavier 初始化，参数也要迁移到GPU
    E = nn.Parameter(torch.empty(Y.shape[0], A.shape[0]).to(device))
    nn.init.xavier_uniform_(E)

    # 定义优化器
    optimizer = torch.optim.Adam([E], lr=0.005)

    # 进行优化
    for epoch in range(1000):
        optimizer.zero_grad()  # 梯度清零
        loss = F.mse_loss(Y, torch.matmul(E, A))  # 计算目标函数
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        E.data = torch.clamp(E.data, min=0)  # 强制 E 非负

    # 获取最终的 E
    E_solution = E.data.cpu().numpy()  # 将结果迁移到CPU

    return E_solution
