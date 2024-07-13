from torch import nn
import torch


class Model(nn.Module):
    def __init__(self, P, L, H, W, commitment_cost=0.9):
        super(Model, self).__init__()

        self.P = P
        self.Channel = L
        act_val = 0.00
        self.encoder_z = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=0.5),
            nn.LeakyReLU(act_val),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=0.5),
            nn.LeakyReLU(act_val),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        )

        h = H // 4
        w = W // 4

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(32 * h * w, h * w),
            nn.BatchNorm1d(h * w),
            nn.Sigmoid(),
            nn.Linear(h * w, L),
            nn.BatchNorm1d(L),
            nn.Sigmoid(),
            nn.Linear(L, P),
            nn.BatchNorm1d(P),
            nn.Sigmoid(),
            nn.Linear(P, P),
        )

        # encoder a
        self.encoder_a = nn.Sequential(
            nn.Conv2d(L, 32 * P, kernel_size=1),
            nn.BatchNorm2d(32 * P),
            nn.LeakyReLU(act_val),
            nn.Conv2d(32 * P, 16 * P, kernel_size=1),
            nn.BatchNorm2d(16 * P),
            nn.LeakyReLU(act_val),
            nn.Conv2d(16 * P, 4 * P, kernel_size=1),
            nn.BatchNorm2d(4 * P),
            nn.LeakyReLU(act_val),
            nn.Conv2d(4 * P, 4 * P, kernel_size=1),
            nn.BatchNorm2d(4 * P),
            nn.LeakyReLU(act_val),
            nn.Conv2d(4 * P, 1 * P, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, y_to_e, y_to_a):
        x = self.encoder_z(y_to_e)
        x = x.reshape(self.Channel, -1)
        e_hat = self.decoder(x)

        a = self.encoder_a(y_to_a)
        a_hat = a.reshape([self.P, -1])
        a_hat = a_hat.squeeze(0)

        y_hat = e_hat @ a_hat

        return y_hat, a_hat, e_hat
