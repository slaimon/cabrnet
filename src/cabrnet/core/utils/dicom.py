import os
import math
import pydicom
import torch
import torch.nn.functional as f

from torch.utils.data import TensorDataset

def load_dicom_dataset(path: str):
    """
    Load a dataset of DICOM (.dcm) files from disk. The directory should be structured thus:
    - root
        - class0
            files...
        - class1
            files...
        - ...

    Args:
        path: Path to root directory of the dataset

    Returns:
        A pytorch TensorDataset where labels are single element tensors containing the class number.
        If the shape of the largest DICOM in the dataset is [H,W,D] then the shape of the samples tensor will be
        [N,H,W,D] and the labels tensor will be [N,1] where N is the number of samples.

    """

    path = path + '/' if path[-1] != '/' else path

    classes = []
    for entry in os.scandir(path):
        if entry.is_dir():
            classes.append(entry.name)

    # load the dataset from disk
    data = []
    labels = []
    for i,label in enumerate(classes):
        for entry in os.scandir(path + label):
            _, ext = os.path.splitext(entry.name)
            if ext == ".dcm":
                # open the DICOM file and convert it to tensor
                print("loading file " + path + label + '/' + entry.name)
                dcm_img = pydicom.dcmread(entry.path)
                npy_img = dcm_img.pixel_array
                data.append(torch.from_numpy(npy_img))
                labels.append(torch.tensor(i))

    # get shape of each sample tensor
    shapes = [x.shape for x in data]

    # get the shape of the smallest tensor that can contain all samples
    max_shape = [max(shapes, key=lambda x: x[i])[i] for i in range(3)]
    print(f"max shape: {max_shape}")

    # apply padding to sample tensors
    for i,x in enumerate(data):
        # compute left and right padding for each dimension
        padding = [max_shape[d] - x.shape[d] for d in range(3)]
        padding = [[math.floor(padding[d]/2), math.ceil(padding[d]/2)] for d in range(3)]

        # flatten the padding list in reverse order to match the expected format for F.pad
        padding = [padding[2][0], padding[2][1], padding[1][0], padding[1][1], padding[0][0], padding[0][1]]

        # apply padding
        padded_x = f.pad(x, padding).unsqueeze(0).expand(3,-1,-1,-1)
        print(f"converted sample #{i+1} from {x.shape} to {padded_x.shape}")
        data[i] = padded_x

    # create dataset
    data = torch.stack(data, dim=0)
    labels = torch.stack(labels, dim=0)

    return TensorDataset(data, labels)