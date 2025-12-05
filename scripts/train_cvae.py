import sys
import os
import torch
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.append(root_dir)

from models.cvae import ConditionalConvVAE as ConditionalVAE



# ----------------- 超参数 -----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 60
BATCH_SIZE = 128
LATENT_DIM = 32
INPUT_DIM = 28 * 28
NUM_CLASSES = 10
LR = 2e-4        # 更稳定的学习率
MODEL_SAVE_PATH = os.path.join(root_dir, "cvae_mnist.pth")


# ----------------- MNIST -----------------
transform = transforms.Compose([
    transforms.ToTensor(),
])

data_path = os.path.join(root_dir, "data")
mnist_dataset = datasets.MNIST(
    root=data_path,
    train=True,
    download=True,
    transform=transform
)
train_loader = DataLoader(mnist_dataset, batch_size=BATCH_SIZE, shuffle=True)


model = ConditionalVAE(
    latent_dim=LATENT_DIM,
    num_classes=NUM_CLASSES
).to(DEVICE)


optimizer = optim.Adam(model.parameters(), lr=LR)


# ----------------- KL annealing 函数 -----------------
def kl_anneal(epoch, total_epoch):
    # 经典 sigmoid KL 策略：前期非常弱，后期增强
    return float(1 / (1 + torch.exp(torch.tensor(-(epoch - total_epoch * 0.5)))))


# ----------------- 训练主循环 -----------------
print("Start training CVAE...")

for epoch in range(1, EPOCHS + 1):

    model.train()
    train_loss = 0
    kl_weight = kl_anneal(epoch, EPOCHS)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

    for data, labels in pbar:
        data = data.to(DEVICE)
        labels_onehot = one_hot(labels, num_classes=NUM_CLASSES).float().to(DEVICE)

        optimizer.zero_grad()
        recon, mu, logvar = model(data, labels_onehot)

        # BCE 重构损失
        bce_loss = torch.nn.functional.binary_cross_entropy(
            recon, data, reduction="sum"
        )

        # KL
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss = bce_loss + kl_weight * kl_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pbar.set_postfix(loss=loss.item() / len(data), KL=kl_weight)

    print(f"Epoch {epoch}, AvgLoss={train_loss / len(train_loader.dataset):.4f}")


# ----------------- 保存模型 -----------------
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved at {MODEL_SAVE_PATH}")
