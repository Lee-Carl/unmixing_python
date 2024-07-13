import torch.nn as nn


class Model(nn.Module):
    def __init__(self, P, L, beta):
        super(Model, self).__init__()
        cls = [128, 64, 16]
        self.encoder = nn.Sequential(
            # 1
            nn.Conv2d(L, cls[0], kernel_size=(1, 1), stride=1, padding=(0, 0)),
            nn.BatchNorm2d(cls[0], momentum=0.9),
            nn.LeakyReLU(0),
            # 2
            nn.Conv2d(cls[0], cls[1], kernel_size=(1, 1), stride=1, padding=(0, 0)),
            nn.BatchNorm2d(cls[1], momentum=0.9),
            nn.LeakyReLU(0),
            nn.Conv2d(cls[1], cls[2], kernel_size=(1, 1), stride=1, padding=(0, 0)),
            nn.BatchNorm2d(cls[2], momentum=0.9),
            nn.LeakyReLU(0),
            nn.Conv2d(cls[2], P, kernel_size=(1, 1), stride=1, padding=(0, 0)),
            nn.BatchNorm2d(P, momentum=0.9),
        )

        self.decoder1 = nn.Sequential(
            nn.Conv2d(P, L, kernel_size=1, stride=1, bias=False),
            nn.LeakyReLU(0.00),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(P, L, kernel_size=1, stride=1, bias=False),
            nn.LeakyReLU(0.00),
        )

    def forward(self, x):
        abu_est1 = self.encoder(x).clamp_(0, 1)
        # abu_est1 = self.encoder2(abu_est1)
        re_result1 = self.decoder1(abu_est1)

        abu_est2 = self.encoder(re_result1).clamp_(0, 1)
        # abu_est2 = self.encoder2(abu_est2)
        re_result2 = self.decoder2(abu_est2)

        return abu_est1, re_result1, abu_est2, re_result2
