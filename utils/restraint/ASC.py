import torch
import torch.nn as nn


class ASC(nn.Module):  # 和为1约束
    def __init__(self):
        super(ASC, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.one = torch.tensor(1, dtype=torch.float, device=device)
        # self.register_buffer('one', torch.tensor(1, dtype=torch.float)).to(device)
        self.loss = nn.L1Loss(size_average=False).to(device)
        self.loss = nn.L1Loss(reduction='sum').to(device)

    def get_target_tensor(self, input):
        target_tensor = self.one
        return target_tensor.expand_as(input)

    def forward(self, a, beta=1e-7):
        a = torch.sum(a, 1)
        target_tensor = self.get_target_tensor(a)
        loss = self.loss(a, target_tensor)
        return beta * loss
