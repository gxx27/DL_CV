# 华中科技大学《深度学习与计算机视觉》课程作业

## work1：单隐藏层神经网络的训练
visualize.py: make_moon数据集可视化

numpy_version.py: 不使用Pytorch搭建一个单隐藏层神经网络

pytorch_version.py: 使用Pytorch框架搭建一个单隐藏层神经网络

## work2：打造自己的 MNIST-GAN
visualize.py: MNIST数据集可视化

model_origin.py: 优化前GAN神经网络模型

model.py: 优化后GAN神经网络模型

train.py: 训练函数

test.py: 测试函数 找出生成图片最接近的数据样本

linearly.py: 线性插值

## work3：CIFAR-ViT

amp.py: 使用混合精度训练

linear_attention.py: 使用线性注意力机制优化原有多头自注意力操作

train_vit.py: ViT训练函数

vision_transformer.py: ViT模型

advanced_version.py: 使用DDP+AMP加速模型训练, 并将[3,32,32]->[3,224,224]的图片以改善测试精度

run.sh: 使用多卡训练的脚本