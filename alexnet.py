import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint
# Depending on your Brevitas version, you can also directly pass bit_width=... to the layers:
# e.g. qnn.QuantConv2d(..., weight_bit_width=num_bits, act_bit_width=num_bits, ...)

class QuantAlexNet(nn.Module):
    """
    A slightly adapted AlexNet-like architecture for 28x28 MNIST images.
    Uses Brevitas quantization layers (QuantConv2d, QuantLinear, QuantReLU).
    """
    def __init__(self, num_bits=8, num_classes=10):
        super(QuantAlexNet, self).__init__()
        
        # Instead of specifying explicit quant module wrappers, you can directly pass
        # weight_bit_width=num_bits and act_bit_width=num_bits to Brevitas layers.
        # We'll do it this way for simplicity:
        
        self.features = nn.Sequential(
            # 1st Conv
            qnn.QuantConv2d(
                in_channels=1, out_channels=64, kernel_size=11, stride=4, padding=2,
                weight_bit_width=num_bits, bias=False
            ),
            qnn.QuantReLU(bit_width=num_bits),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 2nd Conv
            qnn.QuantConv2d(
                in_channels=64, out_channels=192, kernel_size=5, padding=2,
                weight_bit_width=num_bits, bias=False
            ),
            qnn.QuantReLU(bit_width=num_bits),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 3rd Conv
            qnn.QuantConv2d(
                in_channels=192, out_channels=384, kernel_size=3, padding=1,
                weight_bit_width=num_bits, bias=False
            ),
            qnn.QuantReLU(bit_width=num_bits),
            
            # 4th Conv
            qnn.QuantConv2d(
                in_channels=384, out_channels=256, kernel_size=3, padding=1,
                weight_bit_width=num_bits, bias=False
            ),
            qnn.QuantReLU(bit_width=num_bits),
            
            # 5th Conv
            qnn.QuantConv2d(
                in_channels=256, out_channels=256, kernel_size=3, padding=1,
                weight_bit_width=num_bits, bias=False
            ),
            qnn.QuantReLU(bit_width=num_bits)
            # We omit the final max-pool from classic AlexNet because it would collapse MNIST's 1x1 to 0x0.
        )
        
        # After the 5th convolution, the spatial dimension is (likely) 1x1 => flattened dimension = 256
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            qnn.QuantLinear(256, 4096, weight_bit_width=num_bits, bias=False),
            qnn.QuantReLU(bit_width=num_bits),
            
            nn.Dropout(p=0.5),
            qnn.QuantLinear(4096, 4096, weight_bit_width=num_bits, bias=False),
            qnn.QuantReLU(bit_width=num_bits),
            
            qnn.QuantLinear(4096, num_classes, weight_bit_width=num_bits, bias=False)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    
    print(f"Train Epoch: {epoch}  Loss: {epoch_loss:.4f}  Accuracy: {epoch_acc:.2f}%")

def test(model, device, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def main(num_bits=1, epochs=20, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # MNIST Dataloaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # standard MNIST normalization
    ])
    
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # Initialize model
    model = QuantAlexNet(num_bits=num_bits).to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        val_loss, val_acc = test(model, device, test_loader, criterion)
        print(f"Test  Epoch: {epoch}  Loss: {val_loss:.4f}  Accuracy: {val_acc:.2f}%")
        
        # Keep track of best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"best_quant_alexnet_{num_bits}bits.pth")

    print(f"Training complete. Best test accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    # Example usage:
    #   python this_script.py
    #
    # Adjust num_bits, epochs, batch_size, and lr for experimentation.
    main(num_bits=4, epochs=20, batch_size=64, lr=1e-3)

