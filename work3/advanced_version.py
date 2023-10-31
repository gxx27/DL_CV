import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from vision_transformer import ViT
import os

local_rank = int(os.environ['LOCAL_RANK'])

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training ViT")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_threads", type=int, default=8)
    parser.add_argument("--n_devices", type=int, default=1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)
    
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl')
    device = torch.device('cuda', local_rank)
    print(local_rank)    
      
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])    

    trainset = datasets.CIFAR10('datasets', train=True, download=False, transform=transform_train)
    trainloader = DataLoader(trainset, sampler=DistributedSampler(trainset), batch_size=args.batch//args.n_devices, num_workers=args.n_threads, drop_last=True)
    testset = datasets.CIFAR10('datasets', train=False, download=False, transform=transform_test)
    testloader = DataLoader(testset, sampler=DistributedSampler(testset), batch_size=args.batch//args.n_devices, num_workers=args.n_threads, drop_last=True)
    
    input_size = trainset[0][0].size(-1)
    batch_size = args.batch//args.n_devices
    lr = args.lr
    dropout_ratio = args.dropout
    
    model = ViT(
        input_size=input_size,
        batch_size=batch_size,
        dropout_ratio=dropout_ratio,
        num_class=10,
    ).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    scaler = GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    train_loss = []
    for epoch in range(args.epoch):
        model.train()
        epoch_loss = 0
        for batch in trainloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)                
                loss = loss_func(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(trainloader)
        train_loss.append(avg_train_loss)
        if local_rank == 0:
            print('Epoch %d, Loss: %.3f' % (epoch+1, loss.item()))

    model.eval()
    correct = 0
    total = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in testloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with autocast():
                outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    if local_rank == 0:
        print('Accuracy of the model on the test set: %.2f %%' % (100 * accuracy))

        # plot
        plt.figure()
        plt.plot(range(len(train_loss)), train_loss, label='Training loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title(f'Training loss on ViT ')
        plt.legend()
        plt.savefig(f'train_loss_amp.png')