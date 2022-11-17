import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100

from tqdm import tqdm
import yaml


from .utils import *

def train_erm(config, model, loader, test_loader, device, fp16_run=False):
    n_epochs = config['n_epochs']
    optim = torch.optim.AdamW(model.parameters(), lr=float(config['lr']), weight_decay=float(config['weight_decay']))
    model.to(device)
    best_acc = 0
    scaler = GradScaler(enabled=fp16_run)
    for epoch in range(n_epochs):
        model.train()
        for img, labels in (pbar := tqdm(loader)):
            img = img.to(device)
            labels = labels.to(device)
            with autocast(enabled=fp16_run):
                output = model(img)
            loss = nn.functional.cross_entropy(output, labels)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            # optim.step()
            scaler.step(optim)
            scaler.update()
            pbar.set_description(f'Epoch: {epoch}, Loss: {loss:.4f}')
        acc = evaluate(model, test_loader, device)
        print(f'Accuracy: {acc}')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'log/erm/{config["model_name"]}_{config["dataset"]}.pth')
    return model

def main():
    with open('config/erm.yaml', 'r') as fr:
        config = yaml.safe_load(fr)
    device = torch.device('cuda:0')
    fp16_run = config['fp16']

    train_loader, test_loader, num_classes = load_dataset(config['dataset'], config['batch_size'])

    model = get_model(model_name=config['model_name'], num_classes=num_classes)
    print('Start training')
    model = train_erm(config, model, train_loader, test_loader, device, fp16_run=fp16_run)

if __name__ == '__main__':
    main()