import torch
import numpy as np

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

def load_torch_dataset(path: str):

    data = torch.load(path+".data.pt")
    labels = torch.load(path+".labels.pt")

    print(f"loading {path} data and labels...")

    # add the channel dimension
    data = data.unsqueeze(1).expand(-1,3,-1,-1,-1)

    print(f"data: {data.shape}\nlabels: {labels.shape}")

    return TensorDataset(data, labels)