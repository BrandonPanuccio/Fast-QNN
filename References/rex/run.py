import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import alexnet

class AlexNet1w1a(nn.Module):
    def __init__(self):
        super(AlexNet1w1a, self).__init__()
        self.model = alexnet(pretrained=False)
        # Adjust the model for CIFAR-10
        self.model.classifier[6] = nn.Linear(4096, 10)
    
    def forward(self, x):
        return self.model(x)

def get_data_loaders(batch_size):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train(model, criterion, optimizer, train_loader, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Training Loss: {epoch_loss:.4f}")

def evaluate(model, criterion, test_loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(test_loader.dataset)
    accuracy = correct / total
    print(f"Validation Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='AlexNet')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--networkversion', type=str, default='1w1a')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()

    if args.network != 'AlexNet' or args.dataset != 'CIFAR10':
        raise ValueError("Only AlexNet with CIFAR10 dataset is supported in this script.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = AlexNet1w1a().to(device)  # Correctly instantiate and move model to device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader, test_loader = get_data_loaders(batch_size=64)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train(model, criterion, optimizer, train_loader, device)
        evaluate(model, criterion, test_loader, device)
        print("-" * 30)

if __name__ == "__main__":
    main()

