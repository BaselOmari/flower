"""Handle basic dataset creation.

In case of PyTorch it should return dataloaders for your dataset (for both the clients
and the server). If you are using a custom dataset class, this module is the place to
define it. If your dataset requires to be downloaded (and this is not done
automatically -- e.g. as it is the case for many dataset in TorchVision) and
partitioned, please include all those functions and logic in the
`dataset_preparation.py` module. You can use all those functions from functions/methods
defined here of course.
"""
"""MNIST dataset utilities for federated learning."""
from typing import List, Optional, Tuple

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from torchvision.datasets import CIFAR10, MNIST

def _download_data(dataset: str) -> Tuple[Dataset, Dataset]:
    DSET = transform = None
    if dataset == 'mnist':
        DSET = MNIST
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
    elif dataset == 'cifar10':
        DSET = CIFAR10
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        ) 
        pass
    else:
        raise Exception('Dataset not supported')

    trainset = DSET("./dataset", train=True, download=True, transform=transform)
    testset = DSET("./dataset", train=False, download=True, transform=transform)
    return trainset, testset


def _shuffle_data(
        dataset: Dataset, 
        seed: int
    ) -> Subset:
    g = torch.Generator()
    g.manual_seed(seed)
    shuffled_idx = torch.randperm(
        len(dataset),
        generator=g
    )
    return Subset(dataset, shuffled_idx)


def _partition_data(
        trainset: Dataset,
        num_clients: int,
        num_classes: int,
        shard_size: int,
        seed:int,
    ) -> List[Dataset]:
    a = shard_size
    d = num_classes
    n = num_clients
    
    trainset = _shuffle_data(trainset, seed)

    classes = [[] for _ in range(d)]
    class_idx = [0 for _ in range(d)]
    for sample in trainset:
        x,y = sample
        classes[y].append(sample)


    client_data = [[] for _ in range(n)]
    for c in range(n//2):
        for idx, cls in enumerate(classes[:d//2]):
            start = (class_idx[idx])*(a//2)
            end = (class_idx[idx]+2)*(a//2)
            class_idx[idx] += 2

            shard = cls[start:end]
            client_data[c].extend(shard)

    c = n//2
    for cls1 in range(d//2):
        for cls2 in range(d//2,d):
            # first half class
            start = (class_idx[cls1])*(a//2)
            end = (class_idx[cls1]+1)*(a//2)
            class_idx[cls1] += 1
            shard1 = cls[start:end]

            # second half class
            start = (class_idx[cls2])*(a//2)
            end = (class_idx[cls2]+4)*(a//2)
            class_idx[cls2] += 4
            shard2 = cls[start:end]

            
            client_data[c].extend(shard1)
            client_data[c].extend(shard2)
            c += 1


    for c, data in enumerate(client_data):
        x,y = zip(*data)
        tensor_x = torch.stack(x)
        tensor_y = torch.Tensor(y)
        tensor_y = tensor_y.type(torch.LongTensor)

        dataset = TensorDataset(tensor_x, tensor_y)
        client_data[c] = dataset
    
    return client_data
    

def load_datasets(
        dataset: str,
        batch_size: Optional[int] = 40,
        seed: Optional[int] = 42,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    num_clients = 50
    trainset, testset = _download_data(dataset)
    a = 196 if dataset == 'mnist' else 68
    trainset_split = _partition_data(trainset, num_clients, 10, a, seed)
    trainloaders = [DataLoader(dset, batch_size=batch_size, shuffle=True) for dset in trainset_split]
    return trainloaders, DataLoader(testset, batch_size=1)
