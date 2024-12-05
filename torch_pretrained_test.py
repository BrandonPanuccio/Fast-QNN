import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import classification_report
import numpy as np
import time
from torch.profiler import profile, record_function, ProfilerActivity
from AlexNetQuant import AlexNetQuant
from ResNetQuant import ResNet50Quant


# Custom weight initialization function
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# Define the datasets and transformations
def get_data_loaders(dataset_name, batch_size=128, validation_split=0.1):
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    elif dataset_name == 'ImageNet':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        raise ValueError("Invalid dataset name. Choose from 'MNIST', 'CIFAR10', 'ImageNet'")

    if dataset_name == 'MNIST':
        dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'ImageNet':
        dataset = torchvision.datasets.ImageNet(root='./data', split='train', download=True, transform=transform)
        testset = torchvision.datasets.ImageNet(root='./data', split='val', download=True, transform=transform)

    # Split dataset into training and validation
    trainloader, valloader = None, None
    if dataset_name in ['MNIST', 'CIFAR10']:
        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        trainset, valset = torch.utils.data.random_split(dataset, [train_size, val_size])

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4)
    elif dataset_name == 'ImageNet':
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        valloader = None

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    return trainloader, valloader, testloader


# Training function for AlexNet and ResNet-50 models on different datasets
def train_model(model, trainloader, valloader, device, epochs=10, learning_rate=0.001, warmup_epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # Add a learning rate scheduler with warm-up
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate * 10,
                                                    steps_per_epoch=len(trainloader), epochs=epochs,
                                                    pct_start=warmup_epochs / epochs)

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Clip gradients to avoid explosion
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            # Step the scheduler
            scheduler.step()

            # Calculate statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Print epoch stats
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(trainloader):.4f}, "
            f"Accuracy: {100 * correct / total:.2f}%"
        )

    print('Finished Training')


# Test function to evaluate a model on a given dataset
def evaluate_model(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(dataloader)

    return {'accuracy': accuracy, 'loss': avg_loss}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")

    datasets = ['MNIST', 'CIFAR10']
    models_to_test = {
        'AlexNetQuant_8w8a': AlexNetQuant(num_classes=10, weight_bit_width=8, act_bit_width=8),
        'ResNet50Quant_8w8a': ResNet50Quant(num_classes=10, weight_bit_width=8, act_bit_width=8)
    }

    for dataset_name in datasets:
        for model_name, model in models_to_test.items():

            trainloader, valloader, testloader = get_data_loaders(dataset_name)
            model = nn.DataParallel(model)
            model = model.to(device)

            # Initialize model weights
            initialize_weights(model)

            output_filename = f"{model_name}_{dataset_name}_evaluation_results.txt"
            with open(output_filename, "w") as f:
                # Training Phase (only if not ImageNet pre-trained model)
                if trainloader and model_name:
                    f.write(f"\nTraining on dataset: {dataset_name}\n")
                    print(f"\nTraining on dataset: {dataset_name}")
                    f.write(f"\nTraining model: {model_name}\n")
                    print(f"\nTraining model: {model_name}")

                    if dataset_name == 'MNIST':
                        train_model(model, trainloader, valloader, device, epochs=50, learning_rate=0.001)
                    elif dataset_name == 'CIFAR10':
                        train_model(model, trainloader, valloader, device, epochs=200, learning_rate=0.01)

                torch.save(model.state_dict(), f"{model_name}_{dataset_name}_trained.pth")

                # Testing Phase
                f.write(f"\nEvaluating model: {model_name}\n")
                print(f"\nEvaluating model: {model_name}")
                metrics = evaluate_model(model, testloader, device)
                torch.save(model.state_dict(), f"{model_name}_{dataset_name}_evaluated.pth")

                f.write(f"Accuracy: {metrics['accuracy']:.2f}%\n")
                f.write(f"Loss: {metrics['loss']:.4f}\n")

                print(f"Accuracy: {metrics['accuracy']:.2f}%")
                print(f"Loss: {metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
