import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model import Generator, Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
HIDDEN_DIM = 128
IMG_DIM = 28 * 28
BATCH_SIZE = 1024
EPOCHS = 100
LR = 2e-4

transform = transforms.Compose([
    transforms.ToTensor(), # 转换为Tensor
    transforms.Normalize((0.5,), (0.5,)) # 归一化
]) # 定义图像预处理的方法
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=False) # 导入图像数据集
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True) # 以DataLoader的方式导入数据集

generator = Generator(HIDDEN_DIM, IMG_DIM)
discriminator = Discriminator(IMG_DIM)

generator.to(device)
discriminator.to(device)

# 定义损失函数和优化器
loss_func = nn.BCELoss()
optimizer_gen = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_dis = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

# 训练过程
train_loss_gen = []
train_loss_dis = []
for epoch in range(EPOCHS):
    train_loss_gen_epoch = 0
    train_loss_dis_epoch = 0
    for i, (imgs, _) in enumerate(train_loader):
        real_imgs = imgs.to(device)
        size = real_imgs.size(0)
        fake_imgs = torch.randn(size=(size, HIDDEN_DIM)).to(device)

        # training discriminator
        optimizer_dis.zero_grad()
        real_output = discriminator(real_imgs) # 真实输出

        loss_real_dis = loss_func(real_output, torch.ones_like(real_output))

        gen_imgs = generator(fake_imgs)
        fake_output = discriminator(gen_imgs.detach())
        
        loss_fake_dis = loss_func(fake_output, torch.zeros_like(fake_output))

        loss_dis = loss_real_dis + loss_fake_dis
        loss_dis.backward()
        optimizer_dis.step()
        train_loss_dis_epoch += loss_dis.item()

        # training generator
        optimizer_gen.zero_grad()
        fake_output = discriminator(gen_imgs)
        loss_gen = loss_func(fake_output, torch.ones_like(fake_output))
        loss_gen.backward()
        optimizer_gen.step()
        train_loss_gen_epoch += loss_gen.item()

    # compute epoch loss
    avg_train_loss_dis_epoch = train_loss_dis_epoch/len(train_loader)
    avg_train_loss_gen_epoch = train_loss_gen_epoch/len(train_loader)

    print(f'epoch {epoch+1} finished!, gen loss is:{avg_train_loss_gen_epoch}, dis loss is:{avg_train_loss_dis_epoch}')
    train_loss_dis.append(avg_train_loss_dis_epoch)
    train_loss_gen.append(avg_train_loss_gen_epoch)

    # 保存生成的图片
    if (epoch+1) % 10 == 0:
        save_image(gen_imgs.data[:25], f"images/{epoch+1}.png", nrow=5, normalize=True) # 生成图片的前25个进行保存

# 保存模型参数
torch.save(generator.state_dict(), './model/generator.pth')
torch.save(discriminator.state_dict(), './model/discriminator.pth')

# 保存训练loss曲线
plt.figure()
plt.plot(range(len(train_loss_dis)), train_loss_dis, label='Discriminator Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss of Discriminator')
plt.legend()
plt.savefig('train_loss_dis.png')

plt.figure()
plt.plot(range(len(train_loss_gen)), train_loss_gen, label='Generator Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss of Generator')
plt.legend()
plt.savefig('train_loss_gen.png')