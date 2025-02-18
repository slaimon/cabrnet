import torch
import numpy as np

from torch.utils.data import TensorDataset

def load_zeros(in_shape: tuple[int], size: int, channels: int=3):
    expansion_shape = (-1,channels) + len(in_shape)*(-1,)
    samples = torch.zeros((size,) + tuple(in_shape)).unsqueeze(1).expand(expansion_shape)
    labels = torch.zeros((size,1))
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