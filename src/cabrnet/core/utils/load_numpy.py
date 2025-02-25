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

def load_numpy_dataset(path: str):
    """
    Load a dataset in numpy format from disk. A numpy dataset is made up of two files in binary, C-order format called
    `"name.samples.np"` and `"name.labels.np"`.

    Args:
        path: Path to the dataset without file extension (e.g. `"path/to/dataset/name"` will read the files
        `path/to/dataset/name.samples.np` and `path/to/dataset/name.labels.np`).

    Returns:
        A pytorch TensorDataset containing the same data as the numpy dataset

    """

    data_file = path + '.samples.npy'
    labels_file = path + '.labels.npy'
    with open(data_file, "rb") as ifp:
        data = np.load(ifp)
    with open(labels_file, "rb") as ifp:
        labels = np.load(ifp)

    # create dataset
    data = torch.from_numpy(data).unsqueeze(1).expand(-1,3,-1,-1,-1)
    labels = torch.from_numpy(labels)

    print(f"data: {data.shape}\nlabels: {labels.shape}")

    return TensorDataset(data, labels)


import os

def load_dataset(path:str):
    path = path + '/' if path[-1]!='/' else path

    classes = []
    for entry in os.scandir(path):
        if entry.is_dir():
            classes.append(entry.name)

    # load the dataset from disk
    data = []
    labels = []
    for i,label in enumerate(classes):
        print(f"loading {label} samples...")
        for entry in os.scandir(path + label):
            _, ext = os.path.splitext(entry.name)
            if ext == ".npy":
                npy_img = np.load(entry.path)
                data.append(npy_img)
                labels.append(i)

    print(f"converting to torch tensors...")
    data = np.array(data)
    # add redundant channels
    data = torch.from_numpy(data).unsqueeze(1).expand(-1,3,-1,-1,-1)
    # transpose the shortest data dimension to dim=2
    data = torch.transpose(data, 2, 4).type(torch.FloatTensor)
    labels = torch.from_numpy(np.array(labels)).type(torch.LongTensor)
    print(f"shape of data: {data.shape}")

    return TensorDataset(data, labels)