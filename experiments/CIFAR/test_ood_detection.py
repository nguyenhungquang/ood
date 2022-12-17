from pathlib import Path
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
    cwd=False
)

import os
import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
from torch.utils.data import Subset
from model_zoo.wrn import WideResNet
from core.detection import Evaluator
from experiments.CIFAR.data import CIFAR10_MEAN, CIFAR10_STD
from experiments.CIFAR.data import get_ood_datasets

def prepair_datasets( in_dataset, DATA_ROOT):
    dset.CIFAR100(DATA_ROOT, train=True, download=True)
    dset.CIFAR100(DATA_ROOT, train=False, download=True)
    dset.CIFAR10(DATA_ROOT, train=True, download=True)
    dset.CIFAR10(DATA_ROOT, train=False, download=True)

    test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    if in_dataset == 'cifar10':
        print('Using CIFAR-10 as typical data')
        train_data = dset.CIFAR10(DATA_ROOT, train=True, transform=test_transform)
        test_data = dset.CIFAR10(DATA_ROOT, train=False, transform=test_transform)
        NUM_CLASSES = 10
    elif in_dataset == 'cifar100':
        print('Using CIFAR-100 as typical data')
        train_data = dset.CIFAR100(DATA_ROOT, train=True, transform=test_transform)
        test_data = dset.CIFAR100(DATA_ROOT, train=False, transform=test_transform)
        NUM_CLASSES = 100
    else:
        raise ValueError
    train_data = Subset(train_data, indices=torch.randperm(len(train_data))[:10000])
    return train_data, test_data, NUM_CLASSES

def get_model(net, args):
    # Restore model
    if args.load != '':
        if 'pretrained' in args.method_name:
            subdir = 'pretrained'
        elif 'oe_tune' in args.method_name:
            subdir = 'oe_tune'
        elif 'energy_ft' in args.method_name:
            subdir = 'energy_ft'
        else:
            subdir = 'oe_scratch'
        ckpts = (Path(args.load) / subdir).glob(f"{args.method_name}_epoch_*.pt")
        ckpts = list(ckpts)
        if len(ckpts) == 0:
            assert False, "Could not find checkpoints" 
        # TODO: Last ckpt
        net.load_state_dict(torch.load(ckpts[-1]))
        print('Model restored! Checkpoint path:', ckpts[-1])

@hydra.main(config_path=root / "experiments" / "CIFAR" / 'configs', config_name='eval.yaml', version_base="1.2")
def main(cfg):
    # torch.manual_seed(1)
    # np.random.seed(1)

    DATA_ROOT = Path(os.environ['TORCH_DATASETS'])
    train_data, test_data, NUM_CLASSES = prepair_datasets(cfg.in_dataset, DATA_ROOT)
    ood_num_examples = len(test_data) // 5
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.test_bs, shuffle=False,
                                            num_workers=cfg.prefetch, pin_memory=True)

    # Create model
    net = WideResNet(cfg.net.layers, NUM_CLASSES, 
                    cfg.net.widen_factor, dropRate=cfg.net.droprate)
    ckpt = torch.load(cfg.ckpt_path)
    ckpt = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    net.load_state_dict(ckpt)
    print('Model restored! Checkpoint path:', cfg.ckpt_path)
    net.eval()
    if cfg.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(cfg.ngpu)))
    if cfg.ngpu > 0:
        net.cuda()
        # torch.cuda.manual_seed(1)

    cudnn.benchmark = True  # fire on all cylinders

    # /////////////// Detection Prelims ///////////////
    detector = hydra.utils.instantiate(cfg.detector)(net)
    # detector = MSPDetector(net)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.test_bs, shuffle=False, 
                                            num_workers=cfg.prefetch, pin_memory=True)
    if detector.require_grad:
        detector.setup(train_loader, num_classes=NUM_CLASSES)
    else:
        with torch.no_grad():
            detector.setup(train_loader, num_classes=NUM_CLASSES)
    # /////////////// Error Detection ///////////////
    # num_right = len(right_score)
    # num_wrong = len(wrong_score)
    # print('\n\nError Detection')
    # show_performance(wrong_score, right_score, method_name=args.method_name)

    # /////////////// OOD Detection ///////////////
    evaluator = Evaluator(detector, ood_num_examples, cfg.test_bs, cfg.num_to_avg)
    evaluator.compute_in_score(test_loader)
    OOD_DATASETS = get_ood_datasets(DATA_ROOT, in_dataset=cfg.in_dataset)
    for ood_name, ood_dataset in OOD_DATASETS.items(): 
        print(f'{ood_name} Detection')
        evaluator.eval_ood(ood_name, ood_dataset)

    # /////////////// Mean Results ///////////////
    print('\nTest Results!!!!!')
    print(evaluator.df)
    if cfg.save_result:
        evaluator.save(cfg.paths.output_dir)
    print('\nMean Test Results')
    print(evaluator.df[['auroc', 'aupr', 'fpr']].mean())

if __name__ == "__main__":
    main()