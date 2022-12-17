import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from densenet import DenseNet3
from wideresnet import WideResNet

def get_model(model_name='resnet50', num_classes=10):
    if model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        d = model.fc.in_features
        model.fc = nn.Linear(d, num_classes)
    elif model_name == 'densenet':
        model = DenseNet3(depth=100, num_classes=num_classes)
    elif model_name == 'wideresnet':
        model = WideResNet(depth=28, num_classes=num_classes, widen_factor=10)
    print(sum([p.numel() for p in model.parameters()]))
    return model

def get_transform(train=True):
    size = [32, 32]
    if train:
        transform = transforms.Compose([
        transforms.RandomResizedCrop(
            size,
            scale=(0.7, 1.0),
            ratio=(1.0, 1.3333333333333333),
            interpolation=2,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transform
    
def load_dataset(dataset, batch_size):
    train_transform = get_transform(train=True)
    test_transform = get_transform(train=False)

    if dataset == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(root='../data', transform=train_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'CIFAR100':
        dataset = torchvision.datasets.CIFAR100(root='../data', transform=train_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR100(root='../data', train=False, transform=test_transform, download=True)
        num_classes = 100
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
    return train_loader, test_loader, num_classes

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    pred, labels = [], []
    for img, label in loader:
        img = img.to(device)
        out = model(img)
        pred.append(out.argmax(dim=1))
        labels.append(label)
    pred = torch.cat(pred).cpu()
    labels = torch.cat(labels)
    acc = (pred == labels).sum() / len(pred)
    return acc