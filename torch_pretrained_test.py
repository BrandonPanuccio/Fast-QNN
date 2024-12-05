import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
import numpy as np
import time
from torch.profiler import profile, record_function, ProfilerActivity
from AlexNetQuant import AlexNetQuant
from ResNetQuant import ResNet50Quant
import matplotlib.pyplot as plt


# Visualize data samples
def visualize_data(trainloader):
    data_iter = iter(trainloader)
    images, labels = next(data_iter)
    print("Sample Labels: ", labels)
    torchvision.utils.save_image(images, "batch_sample.png")
    print("Saved a batch of images to 'batch_sample.png' for inspection.")


# Initialize model weights
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# Define the datasets and transformations
def get_data_loaders(dataset_name, batch_size=64):
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    elif dataset_name == 'ImageNet':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    if dataset_name == 'MNIST':
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'ImageNet':
        trainset = None
        testset = torchvision.datasets.ImageNet(root='./data', split='val', download=True, transform=transform)
    else:
        raise ValueError("Invalid dataset name. Choose from 'MNIST', 'CIFAR10', 'ImageNet'")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4) if trainset else None
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    return trainloader, testloader


# Train the model
def train_model(model, trainloader, device, epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Log batch-wise loss
            print(f"Batch {batch_idx + 1}/{len(trainloader)}, Loss: {loss.item():.4f}")

        scheduler.step()

        # Epoch stats
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(trainloader):.4f}, "
            f"Accuracy: {100 * correct / total:.2f}%"
        )


# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")

    datasets = ['MNIST', 'CIFAR10']
    models_to_test = {
        'AlexNetQuant_1w1a': AlexNetQuant(num_classes=10, weight_bit_width=1, act_bit_width=1),
        'ResNet50Quant_1w1a': ResNet50Quant(num_classes=10, weight_bit_width=1, act_bit_width=1),
    }

    for dataset_name in datasets:
        for model_name, model in models_to_test.items():
            trainloader, testloader = get_data_loaders(dataset_name)
            visualize_data(trainloader)  # Check data pipeline
            model.apply(init_weights)  # Apply weight initialization
            model = nn.DataParallel(model).to(device)

            # Warm-up training for quantized models
            if '1w1a' in model_name:
                print("Starting warm-up training with 4w4a...")
                model.module.weight_bit_width = 4
                train_model(model, trainloader, device, epochs=50, learning_rate=0.001)
                model.module.weight_bit_width = 1

            # Train final model
            train_model(model, trainloader, device, epochs=300, learning_rate=0.0001)
            torch.save(model.state_dict(), f"{model_name}_{dataset_name}_trained.pth")

            # Evaluate model
            metrics = test_model(model, testloader, device)
            print(f"Final Accuracy: {metrics['accuracy']:.2f}%")

if __name__ == "__main__":
    main()
