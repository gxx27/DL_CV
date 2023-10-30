import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from vision_transformer import ViT


device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training ViT")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = datasets.CIFAR10('datasets', train=True, download=False, transform=transform)
    trainloader = DataLoader(trainset, batch_size=args.batch, num_workers=12, shuffle=True)
    testset = datasets.CIFAR10('datasets', train=False, download=False, transform=transform)
    testloader = DataLoader(testset, batch_size=args.batch, num_workers=12, shuffle=False)
    
    input_size = trainset[0][0].size(-1)
    batch_size = args.batch
    lr = args.lr
    dropout_ratio = args.dropout
    
    model = ViT(
        input_size=input_size,
        batch_size=batch_size,
        dropout_ratio=dropout_ratio,
        num_class=10,
    ).to(device)

    scaler = GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    train_loss = []
    for epoch in range(args.epoch):
        model.train()
        epoch_loss = 0
        for batch in tqdm(trainloader):
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
    print('Accuracy of the model on the test set: %.2f %%' % (100 * accuracy))

    # plot
    plt.figure()
    plt.plot(range(len(train_loss)), train_loss, label='Training loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f'Training loss on ViT ')
    plt.legend()
    plt.savefig(f'train_loss_amp.png')    