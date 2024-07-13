import torch


def SRE(a_true, a_pred, beta=1):
    a = torch.norm(a_true, p=2)
    b = torch.norm(a_true - a_pred, p=2)
    sre = 20 * torch.log10(a / b)
    return sre * beta
