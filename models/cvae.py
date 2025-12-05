import torch
from torch import nn
import torch.nn.functional as F


class ConditionalConvVAE(nn.Module):
    def __init__(self, latent_dim=32, num_classes=10):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # -------- 编码器 --------
        # 输入：图像 (1×28×28) + one-hot 标签 → 作为额外通道
        self.encoder = nn.Sequential(
            nn.Conv2d(1 + num_classes, 32, 3, stride=2, padding=1),  # → 32×14×14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),               # → 64×7×7
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        # -------- 解码器 --------
        self.fc_decode = nn.Linear(latent_dim + num_classes, 64 * 7 * 7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),      # → 32×14×14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),       # → 1×28×28
            nn.Sigmoid()                                             # 像素范围 [0,1]
        )

    # ----------- reparameterize -----------
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ----------- encode -----------
    def encode(self, x, y_onehot):
        # y_onehot: shape [B,10] → reshape 成 [B,10,28,28]
        y_channel = y_onehot.view(y_onehot.size(0), self.num_classes, 1, 1)
        y_channel = y_channel.expand(-1, -1, x.size(2), x.size(3))

        # 拼接为 (1+10) 通道
        h = torch.cat([x, y_channel], dim=1)

        h = self.encoder(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    # ----------- decode（生成使用）-----------
    def decode(self, z, y_onehot):
        # z: [B,latent_dim], y_onehot: [B,10]
        inputs = torch.cat([z, y_onehot], dim=1)

        h = self.fc_decode(inputs)
        h = h.view(-1, 64, 7, 7)     # reshape 为卷积 feature map
        x_recon = self.decoder(h)
        return x_recon

    # ----------- forward（训练使用）-----------
    def forward(self, x, y_onehot):
        mu, logvar = self.encode(x, y_onehot)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y_onehot)
        return recon, mu, logvar
