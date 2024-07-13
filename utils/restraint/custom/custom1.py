import torch


class custom1:
    def __init__(self, device=None):
        self.epsilon_spe = 1e-16
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def compute_y(self, y):
        mt = y
        mt_row_nums = mt.shape[0]  # mt矩阵的行数(m矩阵的列数)
        W_spe = torch.diag(torch.tensor([1. / (torch.norm(mt[i], p=2) + self.epsilon_spe) for i in range(mt_row_nums)]))
        return W_spe.to(self.device)

    def __call__(self, y, beta=1):
        """
            y:  L*N
        """
        weight = self.compute_y(y.clone())
        row_norms = torch.norm(weight, p=2, dim=1)
        # 计算 L2,1 范数
        out = torch.norm(row_norms, p=1)
        return out * beta
