import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.nn.functional import cosine_similarity
from model import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_DIM = 128
IMG_DIM = 28 * 28

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=False)

generator = Generator(HIDDEN_DIM, IMG_DIM).to(device)
generator.load_state_dict(torch.load('./model/generator.pth'))

num_samples = 10
z = torch.randn(num_samples, HIDDEN_DIM).to(device) # noise
generated_images = generator(z)

# 寻找与生成图片最接近的训练集图片
true_img = []
for image in generated_images.cpu():
    closest = None
    sim = -1
    for real_image, _ in train_dataset:
        simil = cosine_similarity(image.view(1, -1), real_image.view(1, -1)).item() # 计算余弦相似度
        if simil > sim:
            closest = real_image
            sim = simil
    true_img.append(closest)

for i in range(num_samples):
    fig, axes = plt.subplots(1, 2)

    axes[0].imshow(generated_images[i].squeeze().detach().cpu(), cmap='gray') # 第一个子图中绘制生成的图片
    axes[0].set_title('Generated Image')

    axes[1].imshow(true_img[i].squeeze().detach().cpu(), cmap='gray') # 第二个子图中绘制与生成图片最接近的训练集图片
    axes[1].set_title('Closest Training Image')


    axes[0].axis('off')
    axes[1].axis('off')

    plt.savefig(f'generate/generated_{i}.png')
    plt.close(fig)