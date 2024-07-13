from torch import nn
import torch

act_val = 0.00


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
    def __init__(self, P, L, commitment_cost=0.9):
        super(Model, self).__init__()

        self.P = P
        self.Channel = L

        self.encoder_z = nn.Sequential(
            nn.Conv2d(L, 32 * P, kernel_size=1),
            nn.BatchNorm2d(32 * P),
            nn.LeakyReLU(act_val),
            nn.Conv2d(32 * P, 16 * P, kernel_size=1),
            nn.BatchNorm2d(16 * P),
            nn.LeakyReLU(act_val),
            nn.Conv2d(16 * P, 4 * P, kernel_size=1),
            nn.BatchNorm2d(4 * P),
            nn.LeakyReLU(act_val),
            nn.Conv2d(4 * P, 1 * P, kernel_size=1),
        )

        self.codebook = VectorQuantizer(num_embeddings=L, embedding_dim=P, commitment_cost=commitment_cost)

        # decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(P, L, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(L, L, kernel_size=1),
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

    def forward(self, inputs):
        x = self.encoder_z(inputs)
        quantized, vq_loss, indices = self.codebook(x)
        y_hat = self.decoder(quantized)

        a_hat = self.encoder_a(inputs)  # B,P,Patch,Patch

        e_hat = self.decoder[0].weight.data

        return y_hat, a_hat, e_hat, vq_loss
