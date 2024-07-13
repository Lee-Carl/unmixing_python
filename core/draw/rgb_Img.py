import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def rgb_Img(img, name="RGB", savepath=None):
    # img:[L, H, W]
    y = img[0, :, :]
    plt.imshow(y)
    plt.axis('off')  # 关闭坐标轴
    if savepath:
        plt.savefig(f"{savepath}/{name}.jpg")
    plt.show()
