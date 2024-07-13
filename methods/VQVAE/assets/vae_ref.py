import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset


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


# 定义VQ-VAE模型
class VQVAE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VQVAE, self).__init__()
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
        quantized, vq_loss, indices = self.vq(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, indices


# 定义损失函数
def loss_function(x_recon, x, vq_loss):
    recon_loss = nn.BCELoss(reduction='sum')(x_recon, x)
    return recon_loss + vq_loss


# 加载MNIST数据集
# transform = transforms.Compose([transforms.ToTensor()])
# train_loader = DataLoader(datasets.MNIST('data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)

# 定义随机生成数据的函数
def generate_random_data(size=1000):
    data = torch.randn(size, 1, 28, 28)  # 生成随机数据，这里假设数据是 1 通道，大小为 28x28
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
        data = data[0]
        optimizer.zero_grad()
        recon_batch, vq_loss, _ = vq_vae(data)
        loss = loss_function(recon_batch, data, vq_loss)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


# 训练模型多个epoch
for epoch in range(1, 11):
    train(epoch)
