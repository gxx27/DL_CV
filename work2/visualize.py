from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=False)

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flatten()): # 取前十张图片出来可视化
    img, label = train_dataset[i]
    ax.imshow(img.squeeze().numpy(), cmap='gray')
    ax.set_title(f"Label: {label}")
    ax.axis('off')
    img_path = os.path.join('./mnist_images', "mnist.png")

plt.savefig(img_path)