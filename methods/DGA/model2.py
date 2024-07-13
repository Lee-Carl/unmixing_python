from torch import nn
import torch
import torch.functional as F
from torch.nn import Module, Sequential, Linear, Conv2d, \
    LeakyReLU, Sigmoid, MaxPool2d, ConvTranspose2d, \
    BatchNorm1d, BatchNorm2d, ELU, SELU, Dropout, Softmax

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
act_val = 0.01


class PGMSU2(Module):
    def __init__(self, P, Channel, z_dim):
        super(PGMSU2, self).__init__()
        self.P = P
        self.Channel = Channel
        # encoder z  fc1 -->fc5
        self.fc4 = Linear(4 * P, z_dim)
        self.fc5 = Linear(4 * P, z_dim)

        # encoder a
        self.encoder_a1 = Sequential(
            Linear(Channel, 32 * P),
            BatchNorm1d(32 * P),
            LeakyReLU(act_val),
            Linear(32 * P, 16 * P),
            BatchNorm1d(16 * P),
            LeakyReLU(act_val),
        )

        self.skip_connect1 = Sequential(
            Linear(Channel, 16 * P),
            BatchNorm1d(16 * P),
            LeakyReLU(act_val),
        )

        self.encoder_a2 = Sequential(
            Linear(16 * P, 8 * P),
            BatchNorm1d(8 * P),
            LeakyReLU(act_val),
            Linear(8 * P, 4 * P),
            BatchNorm1d(4 * P),
            LeakyReLU(act_val),
        )

        self.skip_connect2 = Sequential(
            Linear(16 * P, 4 * P),
            BatchNorm1d(4 * P),
            LeakyReLU(act_val),
        )

        self.encoder_a3 = Sequential(
            LeakyReLU(act_val),
            Linear(4 * P, 1 * P),
            Softmax(dim=1)
        )

        # decoder
        self.decoder = Sequential(
            Linear(z_dim, P * 4),
            BatchNorm1d(P * 4),
            LeakyReLU(act_val),
            Linear(P * 4, P * 64),
            BatchNorm1d(P * 64),
            LeakyReLU(act_val),
            Linear(P * 64, Channel * P),
            Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = (log_var * 0.5).exp()
        eps = torch.randn(mu.shape, device=device)
        return mu + eps * std

    def forward(self, inputs):
        x1 = self.encoder_a1(inputs)
        x1 += self.skip_connect1(inputs)
        x2 = self.encoder_a2(x1)
        x2 += self.skip_connect2(x1)
        mu = self.fc4(x2)
        log_var = self.fc5(x2)
        a = self.encoder_a3(x2)

        # reparameterization trick
        z = self.reparameterize(mu, log_var)
        em = self.decoder(z)

        em_tensor = em.view([-1, self.P, self.Channel])
        a_tensor = a.view([-1, 1, self.P])
        y_hat = a_tensor @ em_tensor
        y_hat = torch.squeeze(y_hat, dim=1)

        return y_hat, mu, log_var, a, em_tensor
