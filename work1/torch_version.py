import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Model, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activate(x)
        output = self.fc2(x)
        
        return output

if __name__ == '__main__':
    x, y = make_moons(n_samples=1000, noise=0.1, random_state=42)

    # hyperparameters
    BATCH_SIZE = 16
    LR = 1e-3
    EPOCH = 50
    INPUT_DIM = 2
    HIDDEN_DIM = 16
    OUTPUT_DIM = 2
    
    x, y = make_moons(n_samples=1000, noise=0.1, random_state=42)

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = Model(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_loss = []
    for epoch in range(EPOCH):
        epoch_loss = 0.0
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        train_loss.append(epoch_loss)
        print(f"Epoch {epoch+1}/{EPOCH}, Train Loss: {epoch_loss:.4f}")

    plt.figure()
    plt.plot(train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('training_loss_torch.png')

    model.eval()
    predict = []
    with torch.no_grad():
        test_loss = 0.0
        correct = 0
        total = 0
        for data, labels in test_loader:
            outputs = model(data)
            loss = loss_func(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predict.append(predicted.item())

        test_loss /= len(test_loader)
        test_accuracy = 100 * correct / total
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    x_test = x_test.numpy()
    predicted_labels = np.array(predict)

    plt.figure()
    plt.scatter(x_test[:, 0], x_test[:, 1], c=predicted_labels)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Test Results')
    plt.savefig('torch_result.png')