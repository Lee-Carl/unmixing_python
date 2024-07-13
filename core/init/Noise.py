import numpy as np


class Noise:
    @staticmethod
    def noise1(img, sigma, tip=True):
        """
        作用：增加噪声
        说明：使用了SUnCNN中的方法；这种方法适合手动调整参数
            mean 代表均值
            sigma 代表标准差
        img_pure = np.clip(img, 0, 1)   img的预处理参考；最大值0，最小值1 将图像的像素值限制在0和1之间，以确保像素值的范围正确
        """
        img_noisy = img + np.random.normal(scale=sigma, size=img.shape)  # 添加高斯噪声
        Noise.showtip(img, img_noisy, tip)
        return img_noisy

    @staticmethod
    def noise2(img, snr, tip=True):
        # 作用：增加噪声
        # 说明：复现了ALMM方法中的增加噪声方法；img需要是二维的；这种方法适合直接输入噪声大小
        pw_signal = np.linalg.norm(img, 'fro') ** 2 / np.prod(img.shape)
        npower = pw_signal / (10 ** (snr / 10))
        img_noise = img + np.sqrt(npower) * np.random.randn(*img.shape)
        Noise.showtip(img, img_noise, tip)
        return img_noise

    @staticmethod
    def computeSNR(img_true, img_noise):
        # 作用：计算SNR
        a = np.mean(img_noise ** 2, dtype=np.float64)
        b = np.mean((img_true - img_noise) ** 2, dtype=np.float64)
        SNR = 10 * np.log10(a / b)
        return SNR

    @staticmethod
    def showtip(img, img_noise, tip):
        print(f"SNR={Noise.computeSNR(img, img_noise)}DB") if tip else None


if __name__ == '__main__':
    add = Noise()
    img = np.random.random((3, 6))
    img_pure = np.clip(img, 0, 1)  # 最大值0，最小值1 将图像的像素值限制在0和1之间，以确保像素值的范围正确
    add.noise2(img_pure, snr=0)
