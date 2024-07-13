import matplotlib.pyplot as plt


def vs_endmembers(edm1, edm2, name="vs_endmembers", savepath=None, show=None):
    # note:端元对比图；前提是，端元必须是L * P
    P = edm1.shape[1]
    plt.rcParams['font.size'] = 24
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    for i in range(P):
        plt.figure(figsize=(8, 6))
        """
            color: 标注颜色
            label: 图例名称
        """
        # plt.plot(edm1[:, i], color="#FF0000", label="true")
        # plt.plot(edm2[:, i], color="#0000FF", label="pred")
        # plt.legend()  # 添加图例

        plt.plot(edm1[:, i], color="#FF0000", linestyle='-.')
        plt.plot(edm2[:, i], color="#0000FF")
        # 1
        # plt.xlabel("Band")
        # plt.ylabel("Reflectance")
        # plt.xlim(0,162)
        plt.ylim(0, 1.0)
        plt.yticks([i / 10.0 for i in range(0, 10 + 1, 2)])
        plt.xticks([i for i in range(0, 160 + 1, 20)])
        # 2
        plt.xlabel("波段")
        plt.ylabel("反射率")
        plt.tight_layout()
        if savepath:
            plt.savefig(savepath + f"/{name}-{i}.jpg")
        if show:
            plt.show()
