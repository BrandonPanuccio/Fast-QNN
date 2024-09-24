import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from AlexNetMNIST import AlexNet_1W1A  # Import your model
import brevitas.onnx as bo

# Set device (use 'cuda' if you have a GPU, otherwise 'cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations for the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./../data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='./../data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load your model
model = AlexNet_1W1A().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(num_epochs=5):
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize the model

            running_loss += loss.item()
            print(f'Train Epoch: {epoch} [{i * len(inputs)}/{len(train_loader.dataset)} '
                  f'({100. * i / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            if i == 25:
                break

    print('Finished Training')

# Testing function
def test_model():
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if total > 25:
                break

    print(f'Accuracy of the network on the test images: {100 * correct / total:.2f}%')

# Export the trained model to QONNX using Brevitas
def export_to_qonnx(export_path='alexnet_mnist_qonnx.onnx'):
    model.eval()  # Set model to evaluation mode before exporting
    dummy_input = torch.randn(1, 1, 28, 28).to(device)  # MNIST has 1 channel, 28x28 images
    bo.export_qonnx(model, input_t=dummy_input, export_path=export_path, opset_version=9)
    print(f'Model exported to {export_path}')

# Train the model for 5 epochs
train_model(num_epochs=1)

# Test the model
test_model()
torch.save(model.state_dict(), 'outputs/alexnet.pth')
# Export the model to QONNX format
export_to_qonnx(export_path='outputs/alexnet_1w1a_mnist_qonnx.onnx')
