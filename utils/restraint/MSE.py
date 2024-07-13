import torch


def MSE(inputs, beta=1):
    # torch.numel(inputs) 求维度之积
    mse = (inputs ** 2).sum() / torch.numel(inputs)
    return mse * beta
