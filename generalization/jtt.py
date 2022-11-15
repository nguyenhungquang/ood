import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torchvision

from tqdm import tqdm
import yaml


from .utils import *

def train_stage_1(config, model, loader, device, fp16_run=False):
    n_epochs = config['stage_1']['n_epochs']
    optim = torch.optim.AdamW(model.parameters(), lr=float(config['stage_1']['lr']), weight_decay=float(config['stage_1']['weight_decay']))
    model.to(device)
    model.train()
    scaler = GradScaler(enabled=fp16_run)
    for epoch in range(n_epochs):
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
    torch.save(model.state_dict(), 'log/jtt/model_stage_1.pth')
    return model

def train_stage_2(config, old_model, cur_model, loader, device, fp16_run=False):
    lmbda = config['stage_2']['lmbda']
    n_epochs = config['stage_2']['n_epochs']

    optim = torch.optim.AdamW(cur_model.parameters(), lr=float(config['stage_2']['lr']), weight_decay=float(config['stage_2']['weight_decay']))
    old_model.to(device)
    cur_model.train()
    cur_model.to(device)
    scaler = GradScaler(enabled=fp16_run)
    for epoch in range(n_epochs):
        for img, labels in (pbar := tqdm(loader)):
            img = img.to(device)
            labels = labels.to(device)
            with autocast(enabled=fp16_run):
                logits = old_model(img)
                predicted_labels = logits.argmax(dim=1)
                correct = (predicted_labels == labels).float()
                logits = cur_model(img)
            loss = nn.functional.cross_entropy(logits, labels, reduction='none')
            assert loss.shape == correct.shape
            new_loss = (1 - correct) * loss * lmbda + correct * loss
            new_loss = new_loss.mean()
            optim.zero_grad()
            scaler.scale(new_loss).backward()
            scaler.unscale_(optim)
            # optim.step()
            scaler.step(optim)
            scaler.update()
            pbar.set_description(f'Epoch: {epoch}, Loss: {new_loss:4f}')
    torch.save(cur_model.state_dict(), 'log/jtt/model_stage_2.pth')
    return cur_model

def train():
    with open('config/jtt.yaml', 'r') as fr:
        config = yaml.safe_load(fr)
    device = torch.device('cuda:0')
    fp16_run = config['fp16']

    train_transform = get_transform(train=True)
    test_transform = get_transform(train=False)

    if config['dataset'] == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(root='../data', transform=train_transform, download=True)
        num_classes = 10
    train_loader = DataLoader(dataset, batch_size=64, num_workers=8, pin_memory=True, shuffle=True)

    model_1 = get_model(model_name=config['model_name'], num_classes=num_classes)
    print('Train stage 1')
    model_1 = train_stage_1(config, model_1, train_loader, device, fp16_run=fp16_run)
    # model_1.load_state_dict(torch.load('log/model_stage_1.pth'))
    model_2 = get_model(model_name=config['model_name'], num_classes=num_classes)
    print('Train stage 2')
    model_2 = train_stage_2(config, model_1, model_2, train_loader, device, fp16_run=fp16_run)

if __name__ == '__main__':
    train()