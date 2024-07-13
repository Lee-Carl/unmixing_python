import torch
import torch.nn as nn


# 定义Vector Quantizer层
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, x):
        # x的形状：(batch_size, channel, height, width)
        print(f'x:{x.shape}')
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
    def __init__(self, L, P, num_embeddings, embedding_dim, commitment_cost):
        super(Model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(L, P * 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(P * 16, P * 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(P * 8, P, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(P, P * 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(P * 8, P * 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(P * 16, L, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, indices = self.vq(z)
        x_hat = self.decoder(quantized)

        return x_hat,vq_loss
