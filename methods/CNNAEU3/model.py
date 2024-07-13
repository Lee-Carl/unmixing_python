from torch import nn
import torch.nn.functional as F


class SENet(nn.Module):

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y


class Model(nn.Module):
    def __init__(self, P, L, dp):
        super(Model, self).__init__()
        self.scale = 3.0
        self.lrelu_params = {
            "negative_slope": 0.02,
            "inplace": False,
        }

        self.p = P
        self.L = L
        cls = [P * 32, P * 16]  # channels
        self.encoder1 = nn.Sequential(
            # 1
            nn.Conv2d(
                self.L,
                cls[0],
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(**self.lrelu_params),
            nn.BatchNorm2d(cls[0]),
            # 2
            nn.Conv2d(cls[0],
                      cls[1],
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.LeakyReLU(**self.lrelu_params),
            nn.BatchNorm2d(cls[1]),
            # nn.Dropout2d(p=0.2),
        )

        self.encoder2 = SENet(channel=cls[1], reduction=16)

        self.encoder3 = nn.Sequential(
            # 3
            nn.Conv2d(cls[1],
                      self.p,
                      kernel_size=1,
                      bias=False),
            nn.LeakyReLU(**self.lrelu_params)
        )

        self.decoder = nn.Conv2d(
            self.p,
            self.L,
            kernel_size=11,
            padding=5,
            bias=False,
        )

    @staticmethod
    def upsample_(x, upscale=2):
        return x[:, :, :, None, :, None] \
            .expand(-1, -1, -1, upscale, -1, upscale) \
            .reshape(x.size(0), x.size(1), x.size(2) * upscale, x.size(3) * upscale)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        abund = F.softmax(e3 * self.scale, dim=1)  # 丰度
        x_hat = self.decoder(abund)  # 像元
        return abund, x_hat
