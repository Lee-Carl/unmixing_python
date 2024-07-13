import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset


def fun(name, x):
    print(f'{name}:{x.shape}')


# 定义Vector Quantizer层
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)  # num_embeddings代表字典容量，embedding_dim代表单个词向量的长度
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)  # 对该权重参数进行均匀分布的初始化

    def forward(self, x):
        # x的形状：(batch_size, channel, height, width)
        # x:(128,256,6,6)
        flattened = x.view(-1, self.embedding_dim)  # (18432,64)
        # 计算每个像素与嵌入的欧氏距离
        # 计算理念：$distances_{i,j}=\Vert flattened_i - embeddings \Vert_2^2$ => a^2+b^2-2ab
        # 形状变化：(n,1) + (m,) => (n,m)
        distances = torch.sum(flattened ** 2, dim=1, keepdim=True) + \
                    torch.sum(self.embeddings.weight ** 2, dim=1) - \
                    2 * torch.matmul(flattened, self.embeddings.weight.t())

        # 找到最近的嵌入向量的索引
        indices = torch.argmin(distances, dim=1).unsqueeze(1)  # 先变成(18432),然后变成(18432,1)

        # 得到最近的嵌入向量
        quantized = self.embeddings(indices).view(
            x.shape)  # (18432,1) op (1,64) =>(18432,1,64),这里的op不是乘法，而是指查询操作，所以最终得到的形状是(18432,1,64)

        # 计算损失
        e_latent_loss = torch.mean((quantized.detach() - x) ** 2)
        q_latent_loss = torch.mean((quantized - x.detach()) ** 2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        return quantized, loss, indices


# 定义VQ-VAE模型
class VQVAE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VQVAE, self).__init__()
        self.flag = 0
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
        )

        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        # Adjust the decoder based on your data dimensions
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, _ = self.vq(z)
        x_recon = self.decoder(quantized)
        if self.flag == 0:
            print(f'{x.shape, z.shape, quantized.shape}')
            self.flag = 1
        return x_recon, vq_loss


import torch.nn.functional as F


def loss_function(x_recon, x, vq_loss):
    recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.view(-1).size(0)  # MSE loss without automatic averaging
    return recon_loss + vq_loss


# 定义随机生成数据的函数
def generate_random_data(size=198):
    data = torch.randn(size, 1, 100, 100)  # 生成随机数据，这里假设数据是 1 通道，大小为 28x28
    return data


# 使用随机生成数据创建 DataLoader
random_data = generate_random_data()
random_dataset = TensorDataset(random_data)
train_loader = DataLoader(random_dataset, batch_size=128, shuffle=True)

# 初始化模型、优化器
vq_vae = VQVAE(num_embeddings=512, embedding_dim=64, commitment_cost=0.25)
optimizer = optim.Adam(vq_vae.parameters(), lr=1e-3)


# 训练模型
def train(epoch):
    vq_vae.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = Variable(data[0])
        optimizer.zero_grad()
        recon_batch, vq_loss = vq_vae(data)
        loss = vq_loss
        loss.backward()
        train_loss += loss.item()
        optimizer.step()


# 训练模型多个epoch
for epoch in range(1, 2):
    train(epoch)
