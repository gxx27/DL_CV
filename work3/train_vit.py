import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from vision_transformer import ViT


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training ViT")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()
    return args

def train_model(trainloader, testloader, lr, input_size, batch_size, dropout_ratio, patch_size, emb_dim, num_head, num_layers):
    model = ViT(
        input_size=input_size,
        batch_size=batch_size,
        dropout_ratio=dropout_ratio,
        num_class=10,
        patch_size=patch_size,
        emb_dim=emb_dim,
        num_head=num_head,
        num_layers=num_layers
    ).to(device)

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
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
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
    plt.savefig(f'train_loss_{patch_size}_{emb_dim}_{num_head}_{num_layers}.png')
    
    return accuracy

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
    
    patch_size = [8, 16, 32]
    emb_dim = [384, 768, 1152]
    num_head = [6, 12, 24]
    num_layers = [10, 12, 14]
    
    default_patch_size = 16
    default_emb_dim = 768
    default_num_head = 12
    default_num_layers = 12
    
    accuracy = train_model(trainloader, testloader, lr, input_size, batch_size, dropout_ratio, default_patch_size, default_emb_dim, default_num_head, default_num_layers)
    
    # Grid search
    # accuracy_exp = []
    # for patch in patch_size:
    #     accuracy = train_model(trainloader, testloader, lr, input_size, batch_size, dropout_ratio, patch, default_emb_dim, default_num_head, default_num_layers)
    #     accuracy_exp.append(accuracy)

    # for emb in emb_dim:
    #     accuracy = train_model(trainloader, testloader, lr, input_size, batch_size, dropout_ratio, default_patch_size, emb, default_num_head, default_num_layers)
    #     accuracy_exp.append(accuracy)

    # for head in num_head:
    #     accuracy = train_model(trainloader, testloader, lr, input_size, batch_size, dropout_ratio, default_patch_size, default_emb_dim, head, default_num_layers)
    #     accuracy_exp.append(accuracy)

    # for layers in num_layers:
    #     accuracy = train_model(trainloader, testloader, lr, input_size, batch_size, dropout_ratio, default_patch_size, default_emb_dim, default_num_head, layers)
    #     accuracy_exp.append(accuracy)
        
    print('experiment results are:', accuracy)