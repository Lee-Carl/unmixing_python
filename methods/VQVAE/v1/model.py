import torch
import torch.nn as nn


class Decoder_NonLinear(nn.Module):
    def __init__(self, L):
        super(Decoder_NonLinear, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(L, L, bias=True),
            nn.Sigmoid(),
            nn.Linear(L, L, bias=True),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


# 定义Vector Quantizer层
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=4, embedding_dim=256, commitment_cost=0.9):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, x):
        # x的形状：(batch_size, channel, height, width)
        flattened = x.view(-1, self.embedding_dim)

        # 计算每个像素与嵌入的欧氏距离
        distances = torch.sum(flattened ** 2, dim=1, keepdim=True) + \
                    torch.sum(self.embeddings.weight ** 2, dim=1) - \
                    2 * torch.matmul(flattened, self.embeddings.weight.t())

        # 找到最近的嵌入向量的索引
        indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # 得到最近的嵌入向量
        quantized = self.embeddings(indices).view(x.shape)

        # 计算损失
        e_latent_loss = torch.mean((quantized.detach() - x) ** 2)
        q_latent_loss = torch.mean((quantized - x.detach()) ** 2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        return quantized, loss, indices


class Model(nn.Module):
    def __init__(self, L, P, commitment_cost=0.9):
        super(Model, self).__init__()
        self.L = L
        self.P = P
        self.encoder = nn.Sequential(
            # 1
            nn.Conv2d(L, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 2
            nn.Conv2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 3
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 4
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 5
            nn.Conv2d(32, P, kernel_size=1, stride=1)
        )

        self.softmax = nn.Softmax(dim=1)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.codebook = VectorQuantizer(L, 10000, commitment_cost)

        self.decoder_linearpart = nn.Sequential(
            nn.Linear(P, L)
        )

        self.decoder_nonlinearpart = Decoder_NonLinear(L)

    def normalization(self, x_latent):
        x_latent = x_latent.abs()
        x_latent = x_latent.t() / x_latent.sum(1)
        x_latent = x_latent.t().float()
        return x_latent

    def forward(self, x):
        z = self.encoder(x)
        a_hat = self.softmax(z)
        quantized, vq_loss, indices = self.codebook(z)
        quantized = quantized.squeeze().reshape(-1,self.P)
        de = self.decoder_linearpart(quantized)
        y_hat = self.decoder_nonlinearpart(de)

        return y_hat, a_hat, vq_loss
