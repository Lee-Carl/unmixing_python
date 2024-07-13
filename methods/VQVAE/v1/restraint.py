import torch
import torch.nn as nn


class Restraint:
    @staticmethod
    def sum_to_one_loss(x, gamma_reg=1.0):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        one = torch.tensor(1, dtype=torch.float)
        loss_func = nn.L1Loss(reduction='sum')  # 使用 reduction='sum' 替代 size_average=False
        x = torch.sum(x, dim=1)
        target_tensor = one.expand_as(x).to(device)
        loss = loss_func(x, target_tensor)

        return gamma_reg * loss
