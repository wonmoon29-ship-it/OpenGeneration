import torch
import matplotlib.pyplot as plt
from torch.nn.functional import one_hot

# 导入卷积版 CVAE
from models.cvae import ConditionalConvVAE as ConditionalVAE


DEVICE = "cpu"
LATENT_DIM = 32
NUM_CLASSES = 10
MODEL_PATH = "cvae_mnist.pth"


# ------------------ 加载模型 ------------------
model = ConditionalVAE(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(DEVICE)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
except FileNotFoundError:
    print(f"模型文件 {MODEL_PATH} 未找到，请先训练模型.")
    exit()

model.eval()


# ------------------ 生成函数 ------------------
@torch.no_grad()
def generate_images(digit, num_images=10):
    """
    根据用户输入的 digit（0-9） 生成 num_images 张图像。
    """
    z = torch.randn(num_images, LATENT_DIM).to(DEVICE)

    labels = torch.tensor([digit] * num_images).to(DEVICE)
    labels_onehot = one_hot(labels, num_classes=NUM_CLASSES).float()

    # 解码器输出形状为 B×1×28×28
    imgs = model.decode(z, labels_onehot)

    # 去掉 channel 维度变为 28×28
    return imgs.squeeze(1).cpu()


# ------------------ 主程序 ------------------
if __name__ == "__main__":
    # ----------- 用户输入数字 -----------
    try:
        digit = int(input("请输入一个要生成的数字 (0-9): "))
        if digit < 0 or digit > 9:
            raise ValueError
    except ValueError:
        print("❌ 输入无效，请输入 0 到 9 的整数。")
        exit()

    # ----------- 生成数字图像 -----------
    print(f"\n正在生成数字 {digit} 的图片...\n")
    images = generate_images(digit, 10)

    # ----------- 展示结果 -----------
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for i in range(10):
        axes[i].imshow(images[i].numpy(), cmap="gray")
        axes[i].axis("off")

    plt.suptitle(f"Generated Images for Digit {digit}", fontsize=16)
    plt.show()
