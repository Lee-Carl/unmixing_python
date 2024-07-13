import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, L, num_embeddings, embedding_dim):
        super(Model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(L, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        )

        self.codebook = nn.Embedding(num_embeddings, embedding_dim)

        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, L, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)

        # 通过 codebook 获取最近邻的离散化量化编码 e 和对应的向量 vq
        e_indices = self._quantize(z)  # 使用你的量化方法来获取最近的离散化编码索引
        e = self.codebook(e_indices)

        # 进行解码操作
        x_hat = self.decoder(e)

        return x_hat, e, e_indices

    def _quantize(self, z):
        # 将 z 调整为二维张量 (batch_size, height*width, channels)
        z_flat = z.view(z.size(0), -1, z.size(1))

        # 计算 z 与每个离散编码的距离
        distances = ((z_flat.unsqueeze(2) - self.codebook.weight.unsqueeze(0)) ** 2).sum(dim=3)

        # 找到最近的离散编码索引
        e_indices = torch.argmin(distances, dim=2)

        return e_indices
