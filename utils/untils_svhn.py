import sys
import numpy
import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_training_dataloader(root, mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of SVHN training dataset
        std: std of SVHN training dataset
        path: path to SVHN training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    svhn_training = torchvision.datasets.SVHN(root='./data/svhn', split='train', download=True, transform=transform_train)
    svhn_training_loader = DataLoader(svhn_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return svhn_training_loader


def get_test_dataloader(root, mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of SVHN test dataset
        std: std of SVHN test dataset
        path: path to SVHN test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: SVHN:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    svhn_test = torchvision.datasets.SVHN(root='./data/svhn', split='test', download=True, transform=transform_test)
    svhn_test_loader = DataLoader(svhn_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return svhn_test_loader


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]