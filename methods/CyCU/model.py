import torch.nn as nn


class Model(nn.Module):
    def __init__(self, P, L):
        super(Model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(L, 128, kernel_size=(1, 1), stride=1, padding=(0, 0)),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=1, padding=(0, 0)),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(),
            nn.Conv2d(64, P, kernel_size=(1, 1), stride=1, padding=(0, 0)),
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
