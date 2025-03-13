import random
from typing import Callable

import torch
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import TensorDataset

def load_rand(in_shape: tuple[int], size: int, channels: int=3):
    expansion_shape = (-1,channels) + len(in_shape)*(-1,)
    samples = torch.rand((size,) + tuple(in_shape)).unsqueeze(1).expand(expansion_shape)
    labels = torch.zeros(size).type(torch.LongTensor)
    labels[size//2:] = 1
    print(f"loaded dataset with samples tensor of shape: {samples.shape}")
    return TensorDataset(samples, labels)

def load_zeros(in_shape: tuple[int], size: int, channels: int=3):
    expansion_shape = (-1,channels) + len(in_shape)*(-1,)
    samples = torch.zeros((size,) + tuple(in_shape)).unsqueeze(1).expand(expansion_shape)
    labels = torch.zeros(size).type(torch.LongTensor)
    print(f"loaded dataset with samples tensor of shape: {samples.shape}")
    return TensorDataset(samples, labels)


def random_rotation(x: torch.Tensor):
    angle = random.randint(0,3)
    return torch.rot90(x, angle, [2,3])

def random_reflection(x: torch.Tensor):
    axes = []
    for i in [1,2,3]:
        axes = axes + [i] if random.random() > 0.5 else axes
    return torch.flip(x, axes)

class Composition:
    def __init__(
        self,
        transforms: list[Callable[[torch.Tensor], torch.Tensor]]
    ):
        self.transforms = transforms

    def apply(self, x:torch.Tensor):
        for t in self.transforms:
            x = t(x)
        return x


class AugmentedTensorDataset(Dataset):
    def __init__(
            self,
            data:torch.Tensor,
            labels:torch.Tensor,
            transform:Composition
    ):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data[item]
        if self.transform:
            sample = self.transform.apply(sample)

        return sample, self.labels[item]


def load_torch_dataset(
        path: str,
        augment: bool,
        RGB: bool=True
):
    data = torch.load(path+".data.pt")
    labels = torch.load(path+".labels.pt")

    # add the channel dimension
    data = data.unsqueeze(1)
    if RGB:
        data = data.expand(-1,3,-1,-1,-1)

    if augment:
        transform = Composition([random_rotation, random_reflection])
    else:
        transform = Composition([])

    labels = labels.type(torch.LongTensor)
    return AugmentedTensorDataset(data, labels, transform)