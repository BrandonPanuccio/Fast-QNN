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


# Define the datasets and transformations
def get_data_loaders(dataset_name, batch_size=64):
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if dataset_name == 'MNIST':
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == 'ImageNet':
        trainset = torchvision.datasets.ImageNet(root='./data', split='train', download=True, transform=transform)
        testset = torchvision.datasets.ImageNet(root='./data', split='val', download=True, transform=transform)
    else:
        raise ValueError("Invalid dataset name. Choose from 'MNIST', 'CIFAR10', 'ImageNet'")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


# Test function to evaluate a model on a given dataset
def test_model(model, testloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    running_loss = 0.0
    total_inference_time = 0.0
    total_images = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            start_time = time.time()
            outputs = model(images)
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            total_images += images.size(0)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(testloader)
    avg_latency = total_inference_time / total_images
    throughput = total_images / total_inference_time
    report = classification_report(all_labels, all_preds, output_dict=True)

    # Measure FLOPs using torch.profiler
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
        with record_function("model_inference"):
            model(images)
    flops = sum([evt.flops for evt in prof.events() if evt.flops is not None])

    metrics = {
        'accuracy': accuracy,
        'loss': avg_loss,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score'],
        'avg_latency': avg_latency,
        'throughput': throughput,
        'flops': flops
    }

    return metrics


# Main function to test AlexNet and ResNet-50 on different datasets
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
    datasets = ['MNIST', 'CIFAR10', 'ImageNet']
    models_to_test = {'AlexNet': models.alexnet(pretrained=True),
                      'ResNet50': models.resnet50(pretrained=True)}

    with open("model_evaluation_results.txt", "w") as f:
        for dataset_name in datasets:
            f.write(f"\nTesting on dataset: {dataset_name}\n")
            print(f"\nTesting on dataset: {dataset_name}")
            trainloader, testloader = get_data_loaders(dataset_name)

            for model_name, model in models_to_test.items():
                f.write(f"\nEvaluating model: {model_name}\n")
                print(f"\nEvaluating model: {model_name}")
                model = nn.DataParallel(model)
                model = model.to(device)
                metrics = test_model(model, testloader, device)

                f.write(f"Accuracy: {metrics['accuracy']:.2f}%\n")
                f.write(f"Loss: {metrics['loss']:.4f}\n")
                f.write(f"Precision: {metrics['precision']:.4f}\n")
                f.write(f"Recall: {metrics['recall']:.4f}\n")
                f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
                f.write(f"Average Latency: {metrics['avg_latency']:.6f} seconds/image\n")
                f.write(f"Throughput: {metrics['throughput']:.2f} images/second\n")
                f.write(f"FLOPs: {metrics['flops']}\n")

                print(f"Accuracy: {metrics['accuracy']:.2f}%")
                print(f"Loss: {metrics['loss']:.4f}")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"F1 Score: {metrics['f1_score']:.4f}")
                print(f"Average Latency: {metrics['avg_latency']:.6f} seconds/image")
                print(f"Throughput: {metrics['throughput']:.2f} images/second")
                print(f"FLOPs: {metrics['flops']}")


if __name__ == "__main__":
    main()
