import torch.nn as nn

class Generator(nn.Module):
    """生成网络"""
    def __init__(self, hidden_dim, img_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, 256), # 输入是HIDDEN_DIM维度的噪点
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, img_dim),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z) # 输出一副图像
        img = img.view(img.size(0), 1, 28, 28)
        return img


class Discriminator(nn.Module):
    """判别器网络"""
    def __init__(self, img_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1) # 输入一副图像
        valid = self.model(img_flat) # 输出yes/no
        return valid