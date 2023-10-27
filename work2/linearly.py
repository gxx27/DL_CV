import torch
from torchvision.utils import save_image
from model import Generator

HIDDEN_DIM = 128
IMG_DIM = 28 * 28

generator = Generator(HIDDEN_DIM, IMG_DIM)
generator.load_state_dict(torch.load('./model/generator.pth'))
generator.eval()

z_dim = HIDDEN_DIM 
z1 = torch.randn(1, z_dim) # 定义两个噪声
z2 = torch.randn(1, z_dim)

num_steps = 20  # 插值点
gen = [] # 生成的图片集合
for step in range(num_steps):
    alpha = step / (num_steps - 1)  # 插值因子
    inter_z = alpha * z1 + (1 - alpha) * z2  # 线性插值
    image = generator(inter_z)
    gen.append(image)

output = torch.cat(gen, dim=3) # 合成一张图
save_image(output, 'interpolated.png')