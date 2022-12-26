from pathlib import Path
import torch
import numpy as np
import torchvision.transforms as trn
import torchvision.datasets as dset
from core.datasets.dataset_utils import ImageFolderOOD
import copy

import PIL.Image as PILImage

CIFAR10_MEAN = [x / 255 for x in [125.3, 123.0, 113.9]]
CIFAR10_STD = [x / 255 for x in [63.0, 62.1, 66.7]]

jigsaw = lambda x: torch.cat((
    torch.cat((torch.cat((x[:, 8:16, :16], x[:, :8, :16]), 1),
               x[:, 16:, :16]), 2),
    torch.cat((x[:, 16:, 16:],
               torch.cat((x[:, :16, 24:], x[:, :16, 16:24]), 2)), 2),
), 1)

speckle = lambda x: torch.clamp(x + x * torch.randn_like(x), 0, 1)
rgb_shift = lambda x: torch.cat((x[1:2].index_select(2, torch.LongTensor([i for i in range(32 - 1, -1, -1)])),
                                 x[2:, :, :], x[0:1, :, :]), 0)
invert = lambda x: torch.cat((x[0:1, :, :], 1 - x[1:2, :, ], 1 - x[2:, :, :],), 0)
pixelate = lambda x: x.resize((int(32 * 0.2), int(32 * 0.2)), PILImage.Resampling.BOX).resize((32, 32), PILImage.Resampling.BOX)


class AvgOfPair(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.shuffle_indices = np.arange(len(dataset))
        np.random.shuffle(self.shuffle_indices)

    def __getitem__(self, i):
        random_idx = np.random.choice(len(self.dataset))
        while random_idx == i:
            random_idx = np.random.choice(len(self.dataset))

        return self.dataset[i][0] / 2. + self.dataset[random_idx][0] / 2., 0

    def __len__(self):
        return len(self.dataset)

class GeomMeanOfPair(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.shuffle_indices = np.arange(len(dataset))
        np.random.shuffle(self.shuffle_indices)

    def __getitem__(self, i):
        random_idx = np.random.choice(len(self.dataset))
        while random_idx == i:
            random_idx = np.random.choice(len(self.dataset))

        return trn.Normalize(CIFAR10_MEAN, CIFAR10_STD)(torch.sqrt(self.dataset[i][0] * self.dataset[random_idx][0])), 0

    def __len__(self):
        return len(self.dataset)

def get_uniform_dataset(num_examples, image_shape=(3, 32, 32)):
    dummy_targets = torch.ones(num_examples)
    ood_data = torch.from_numpy(
        np.random.uniform(size=(num_examples, *image_shape),
                        low=-1.0, high=1.0).astype(np.float32))
    return torch.utils.data.TensorDataset(ood_data, dummy_targets)

def get_avgofpair_dataset(base_ood_dataset):
    return AvgOfPair(base_ood_dataset)

def get_geomeanofpair_dataset(base_ood_dataset):
    return AvgOfPair(base_ood_dataset)

def get_jigsave_dataset(base_ood_dataset):
    return _copy_and_replace_transform(base_ood_dataset, jigsaw)

def get_speckle_dataset(base_ood_dataset):
    return _copy_and_replace_transform(base_ood_dataset, speckle)

def get_pixelate_dataset(base_ood_dataset):
    return _copy_and_replace_transform(base_ood_dataset, pixelate, is_PIL_op=True)

def get_rgbshift_dataset(base_ood_dataset):
    return _copy_and_replace_transform(base_ood_dataset, rgb_shift)

def get_inverted_dataset(base_ood_dataset):
    return _copy_and_replace_transform(base_ood_dataset, invert)

def _copy_and_replace_transform(base_ood_dataset, op, is_PIL_op=False):
    base_ood_dataset = copy.copy(base_ood_dataset)
    if is_PIL_op:
        new_transform = trn.Compose([op, trn.ToTensor(),trn.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    else:
        new_transform = trn.Compose([trn.ToTensor(), op, trn.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    base_ood_dataset.transform = new_transform
    return base_ood_dataset

def get_ood_datasets(DATA_ROOT: Path, in_dataset: str):
    OOD_DATASETS = {
        "SVHN": dset.SVHN(root= DATA_ROOT / 'svhn', split="test",
                        transform=trn.Compose(
                            [trn.ToTensor(), 
                            trn.Normalize(CIFAR10_MEAN, CIFAR10_STD)])),
        "LSUN-C": ImageFolderOOD(root= DATA_ROOT / 'data_ood_detection' / 'LSUN', 
                            transform=trn.Compose([trn.ToTensor(), trn.Normalize(CIFAR10_MEAN, CIFAR10_STD)])),
        "LSUN-R": ImageFolderOOD(root= DATA_ROOT / 'data_ood_detection' / 'LSUN_resize', 
                            transform=trn.Compose([trn.ToTensor(), trn.Normalize(CIFAR10_MEAN, CIFAR10_STD)])),
        "iSUN": ImageFolderOOD(root= DATA_ROOT / 'data_ood_detection' / 'iSUN', 
                            transform=trn.Compose([trn.ToTensor(), trn.Normalize(CIFAR10_MEAN, CIFAR10_STD)])),
        "Imagenet-R": ImageFolderOOD(root= DATA_ROOT / 'data_ood_detection' / 'Imagenet_resize', 
                            transform=trn.Compose([trn.ToTensor(), trn.Normalize(CIFAR10_MEAN, CIFAR10_STD)])),
        "Imagenet-C": ImageFolderOOD(root= DATA_ROOT / 'data_ood_detection' / 'Imagenet', 
                            transform=trn.Compose([trn.ToTensor(), trn.Normalize(CIFAR10_MEAN, CIFAR10_STD)])),
        # "Places365": ImageFolderOOD(root= DATA_ROOT / 'data_ood_detection' / 'place365', 
        #                       transform=trn.Compose([trn.Resize(32), 
        #                                             trn.CenterCrop(32),
        #                                             trn.ToTensor(), 
        #                                             trn.Normalize(CIFAR10_MEAN, CIFAR10_STD)])),
        # "Textures": ImageFolderOOD(root= DATA_ROOT / 'data_ood_detection' / 'textures', 
        #                       transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
        #                                                trn.ToTensor(), trn.Normalize(CIFAR10_MEAN, CIFAR10_STD)]),
    }

    if in_dataset == 'cifar10':
        OOD_DATASETS['CIFAR100'] = dset.CIFAR100(DATA_ROOT, train=False, 
                                                 transform=trn.Compose([trn.ToTensor(), trn.Normalize(CIFAR10_MEAN, CIFAR10_STD)]))
    elif in_dataset == 'cifar100':
        OOD_DATASETS['CIFAR10'] = dset.CIFAR10(DATA_ROOT, train=False, 
                                               transform=trn.Compose([trn.ToTensor(), trn.Normalize(CIFAR10_MEAN, CIFAR10_STD)]))


    return OOD_DATASETS