import torch
import torch.nn.functional as F


def norm_l0(inputs, threshold=0, beta=1):
    count = 0
    for param in inputs:
        count += torch.sum(torch.abs(param) > threshold)
    return count.item() * beta


def norm_l1(inputs, beta=1):
    return torch.norm(inputs, p=1) * beta


def norm_l11(inputs, beta=1):
    return inputs.sum().sum() * beta


def norm_nuclear(inputs, beta=1):  # 核范数
    return torch.norm(inputs, p='nuc') * beta


def norm_l21(inputs, beta=1):
    # 仅适合于二维数据
    row_norms = torch.norm(inputs, p=2, dim=1)
    # 计算 L2,1 范数
    out = torch.norm(row_norms, p=1)
    return out * beta


def norm_l1_2(inputs, beta=1):  # L1/2稀疏正则化
    out = torch.mean(torch.sum(torch.sqrt(torch.abs(inputs)), dim=1))
    return out * beta


def norm_l2(inputs, beta=1):
    out = torch.norm(inputs, p=2)
    return out * beta


def norm_tv(x, beta=1):
    # batch_size, channels, height, width = x.size()
    horizontal_variation = torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    vertical_variation = torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return (horizontal_variation + vertical_variation) * beta


def norm_tv_3d(image, dx=1, dy=1, batch_size=None, beta=1):
    """
    计算图像批次（三维张量）的总变分（TV）正则化项。

    参数：
    image -- 输入的三维PyTorch张量，形状为（P, H, W）。
    dx -- x方向的梯度权重。
    dy -- y方向的梯度权重。
    batch_size -- 批次中图像的数量。如果提供，会在计算梯度时考虑批次大小。

    返回：
    TV_norm -- 图像批次的总变分正则化项。
    """
    if batch_size is None:
        batch_size = image.size(0)

    # 计算x方向和y方向的梯度
    grad_x = F.conv2d(image, torch.tensor([[dx, 0], [0, dx]]), stride=1, padding=0)
    grad_y = F.conv2d(image, torch.tensor([[dy, dy], [0, 0]]), stride=1, padding=0)

    # 计算梯度的绝对值
    grad_x = torch.abs(grad_x)
    grad_y = torch.abs(grad_y)

    # 累加所有图像的梯度之和
    TV_norm = torch.sum(grad_x) + torch.sum(grad_y)

    # 除以批次大小，得到每个图像的平均TV正则化项
    TV_norm /= batch_size

    return TV_norm * beta


def norm_tv_3d2(image, dx=1, dy=1, dz=1, batch_size=None):
    """
    计算图像批次（三维张量）的总变分（TV）正则化项。

    参数：
    image -- 输入的三维PyTorch张量，形状为（P, H, W, D）。
    dx -- x方向的梯度权重。
    dy -- y方向的梯度权重。
    dz -- z方向的梯度权重。
    batch_size -- 批次中图像的数量。如果提供，会在计算梯度时考虑批次大小。

    返回：
    TV_norm -- 图像批次的总变分正则化项。
    """
    if batch_size is None:
        batch_size = image.size(0)

    # 计算x方向、y方向和z方向的梯度
    grad_x = F.conv3d(image, torch.tensor([[dx, 0, 0], [0, dx, 0], [0, 0, dx]]), stride=1, padding=0)
    grad_y = F.conv3d(image, torch.tensor([[0, dy, 0], [dy, dy, 0], [0, 0, dy]]), stride=1, padding=0)
    grad_z = F.conv3d(image, torch.tensor([[0, 0, dz], [0, dy, 0], [dz, 0, 0]]), stride=1, padding=0)

    # 计算梯度的绝对值
    grad_x = torch.abs(grad_x)
    grad_y = torch.abs(grad_y)
    grad_z = torch.abs(grad_z)

    # 累加所有图像的梯度之和
    TV_norm = torch.sum(grad_x) + torch.sum(grad_y) + torch.sum(grad_z)

    # 除以批次大小，得到每个图像的平均TV正则化项
    TV_norm /= batch_size

    return TV_norm
