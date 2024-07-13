import torch
import torch.nn as nn


class MinVolumn:  # 最小体积约束MVC
    def __init__(self, band, num_classes):
        self.band = band  # 波段数
        self.num_classes = num_classes  # 纯净端元数

    def __call__(self, edm, beta=1):
        edm_result = torch.reshape(edm, (self.band, self.num_classes))
        edm_mean = edm_result.mean(dim=1, keepdim=True)
        loss = beta * ((edm_result - edm_mean) ** 2).sum() / self.band / self.num_classes
        return loss
