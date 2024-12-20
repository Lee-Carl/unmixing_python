import numpy as np
from custom_types import HsiDataset, HsiData
from typing import Tuple
from custom_types import HsiPropertyEnum
from sklearn.feature_extraction.image import extract_patches_2d
import torch
from torch.utils.data import DataLoader


class HsiUtil:
    @staticmethod
    def checkHsiDatasetDims(data: HsiDataset):
        """ 将数据转换到指定维度 """
        shape: tuple
        tags: list = ['Y', 'A', 'E']
        for tag in tags:
            if hasattr(data, tag) and HsiPropertyEnum.isExistProperty(tag):
                v = HsiPropertyEnum.get_value(tag)
                hpe = HsiPropertyEnum(v)
                shape = HsiUtil.getShapeForData(hpe, data)
                data[tag] = HsiUtil.changeDims(data[tag], shape)
        return data

    @staticmethod
    def getShapeForData(field: HsiPropertyEnum, data: HsiDataset) -> Tuple:
        """ 得到指定的字段的维度 """
        P, L, N = data.getPLN()
        if field == HsiPropertyEnum.E:
            if len(data.E.shape) == 2:
                return L, P
            else:
                return L, P, N
        elif field == HsiPropertyEnum.A:
            return P, N
        elif field == HsiPropertyEnum.Y:
            return L, N
        raise Exception("未能匹配到对应的数据类型")

    @staticmethod
    def changeDims(data: HsiData, shape: tuple) -> HsiData:
        if data.shape != shape:
            dims = tuple(data.shape.index(e) for e in shape)  # 获取目标shape在data的shape中的次序
            data = data.transpose(dims)  # 转置成目标shape
        return data

    @staticmethod
    def gen_Y(e: HsiData, a: HsiData) -> HsiData:
        """ 通过端元和丰度合成混元数据 """
        # 作用：用端元矩阵与丰度矩阵生成像元矩阵Y
        # 丰度一般是二维的
        # 端元有二维，也有三维的。假设E2=(198,4),E3=（198，4，10**4），A=（4，10**4）
        E, A = e.copy(), a.copy()
        if A.shape[0] > A.shape[1]:
            A = A.T  # A=(4,10**4)

        if len(E.shape) == 2:
            if E.shape[0] < E.shape[1]:
                E = E.T  # E2=(198,4)
            return E @ A  # y=(198,10**4)
        else:
            A = A.T  # A=(10**4,4)
            A = np.expand_dims(A, axis=1)  # A=(10**4,1,4)
            s = E.shape
            s = sorted(enumerate(s), key=lambda x: x[1])  # e=[(1, 4), (0, 198), (2, 10000)]
            E = E.transpose(s[2][0], s[0][0], s[1][0])  # E3=(10**4,4,198)
            Y: HsiData = A @ E  # y=(10**4,1,198)
            Y = np.squeeze(Y, axis=1)  # y=(10**4,198)
            Y = Y.T  # y=(198,10**4)
            return Y

    @staticmethod
    def gen_DataLoader(pixels: HsiData, patch_size: Tuple[int, int], num_patches: int, batch_size: int):
        """ 将像元裁切，并导出一个数据集 """
        # 随机提取原始图像patch大小尺寸图片
        # pixels必须是 L * H * W
        pixels_numpy = pixels.transpose((1, 2, 0))

        input_patches = extract_patches_2d(
            pixels_numpy,
            max_patches=num_patches,
            patch_size=patch_size,
        )

        input_patches = torch.Tensor(input_patches.transpose((0, 3, 1, 2)))

        # Dataloader
        dataloader = DataLoader(
            input_patches,
            batch_size=batch_size,
            shuffle=True,
        )
        return dataloader
