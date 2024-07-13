import torch.nn as nn
from .subModule import attention, SENet


class Model(nn.Module):
    def __init__(self, P, L):
        super(Model, self).__init__()
        cls = [128, 64]
        self.encoder = nn.Sequential(
            nn.Conv2d(L, cls[0], kernel_size=(1, 1), stride=1, padding=(0, 0)),
            nn.BatchNorm2d(cls[0], momentum=0.9),
            nn.LeakyReLU(0),
            nn.Conv2d(cls[0], cls[1], kernel_size=(1, 1), stride=1, padding=(0, 0)),
            nn.BatchNorm2d(cls[1], momentum=0.9),
            nn.ReLU(),
            nn.Conv2d(cls[1], P, kernel_size=(1, 1), stride=1, padding=(0, 0)),
            nn.BatchNorm2d(P, momentum=0.9),
        )

        # self.encoder2 = SENet(cls[1])  # 你敢信，去除这个代码，结果就变了！

        self.encoder3 = nn.Sequential(
            nn.Conv2d(cls[1], P, kernel_size=(1, 1), stride=1, padding=(0, 0)),
            nn.BatchNorm2d(P, momentum=0.9),
        )

        self.decoder1 = nn.Sequential(
            nn.Conv2d(P, L, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(P, L, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
        )

    def forward(self, x):
        abu_est1 = self.encoder(x).clamp_(0, 1)
        re_result1 = self.decoder1(abu_est1)

        abu_est2 = self.encoder(re_result1).clamp_(0, 1)
        re_result2 = self.decoder2(abu_est2)

        return abu_est1, re_result1, abu_est2, re_result2
