import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, channels):
        super(Attention, self).__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        energy = self.conv(x).view(batch_size, -1, height * width)
        attention = self.softmax(energy)

        out = torch.bmm(x.view(batch_size, channels, -1), attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        return out
