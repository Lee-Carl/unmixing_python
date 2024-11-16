import torch


def SAD(e_true, e_pred, beta=1):
    # 1.
    # a = torch.sqrt(torch.sum(e_true ** 2, 0))
    # b = torch.sqrt(torch.sum(e_pred ** 2, 0))
    # sad = torch.acos(torch.sum(e_true * e_pred, dim=0) / a / b).mean()

    # 2.
    a = torch.norm(e_pred, dim=1, p=2)
    b = torch.norm(e_true, dim=1, p=2)
    sad = torch.mean(torch.acos(torch.sum(e_true * e_pred, dim=1) / (a * b)))
    return sad * beta
