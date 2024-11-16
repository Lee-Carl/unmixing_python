import torch


def RMSE(e_true, e_pred, beta=1):
    a = torch.norm(e_true - e_pred, dim=1, p=2)
    return torch.sqrt(torch.mean(torch.pow(a, 2))) * beta


def RMSE2(e_true, e_pred, beta=1):
    rmse = torch.sqrt(torch.sum((e_true - e_pred) ** 2) / torch.prod(torch.tensor(e_true.shape).float()))
    return rmse.item() * beta
