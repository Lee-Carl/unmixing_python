from torch import nn
import torch


class Model(nn.Module):
    def __init__(self, P, L, z_dim, H, W):
        super(Model, self).__init__()
        self.P = P
        self.L = L
        self.H = H
        self.W = W
        self.kernel_size = 1
        self.stride = 1
        self.padding = 0
        self.act_val = 0.01
        self.dp = 0.0
        # encoder a
        self.encoder_a = nn.Sequential(
            nn.Conv2d(L, 32 * P, kernel_size=5, stride=self.stride, padding=2),
            nn.BatchNorm2d(32 * P),
            nn.LeakyReLU(self.act_val),
            nn.Conv2d(32 * P, 16 * P, kernel_size=5, stride=self.stride, padding=2),
            nn.BatchNorm2d(16 * P),
            nn.LeakyReLU(self.act_val),
            nn.AvgPool2d(kernel_size=1, stride=1),
            nn.Conv2d(16 * P, 4 * P, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(4 * P),
            nn.LeakyReLU(self.act_val),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(4 * P, 4 * P, kernel_size=2, stride=2),
            nn.BatchNorm2d(4 * P),
            nn.LeakyReLU(self.act_val),
            nn.ConvTranspose2d(4 * P, 4 * P, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(4 * P),
            nn.LeakyReLU(self.act_val),
            nn.ConvTranspose2d(4 * P, 1 * P, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.Softmax(dim=1)
        )

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(L, 32 * P, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm2d(32 * P),
            nn.LeakyReLU(self.act_val),
            nn.Conv2d(32 * P, 16 * P, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm2d(16 * P),
            nn.LeakyReLU(self.act_val),
            nn.Conv2d(16 * P, 4 * P, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm2d(4 * P),
            nn.LeakyReLU(self.act_val),
        )
        self.fc4 = nn.Conv2d(4 * P, z_dim, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.fc5 = nn.Conv2d(4 * P, z_dim, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, P * 4, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm2d(P * 4),
            nn.LeakyReLU(self.act_val),
            nn.ConvTranspose2d(P * 4, P * 64, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm2d(P * 64),
            nn.LeakyReLU(self.act_val),
            nn.ConvTranspose2d(P * 64, L * P, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.Sigmoid()
        )

    @staticmethod
    def reparameterize(mu, log_var):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        std = (log_var * 0.5).exp()
        eps = torch.randn(mu.shape).to(device)
        return mu + eps * std

    def forward(self, inputs):
        """
            inputs: 1, L, H, W
            x: 1, 4P , H/2, W/2
            fc4/fc5: 1, 4P, H/2, W/2
        """
        x = self.encoder(inputs)
        mu = self.fc4(x)
        log_var = self.fc5(x)
        # reparameterization trick
        z = self.reparameterize(mu, log_var)
        em = self.decoder(z)
        em_tensor = em.view([1, self.P, self.L, self.H, self.W]).squeeze(0).view(
            [self.P, self.L, self.H * self.W]).permute(2, 0, 1)

        a = self.encoder_a(inputs)
        a_tensor = a.view([1, self.P, self.H * self.W]).permute(2, 0, 1)

        y_hat = a_tensor @ em_tensor
        y_hat = torch.squeeze(y_hat, dim=1)

        y_hat = y_hat.reshape(1, -1, self.H, self.W)
        return y_hat, mu, log_var, a, em_tensor
