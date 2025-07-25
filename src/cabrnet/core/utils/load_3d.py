import os

import torch
import random

from PIL import Image
import numpy as np

from loguru import logger
from fractions import Fraction
from typing import Callable

from pyarrow import duration
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset

from torchvision.datasets import Kinetics
from torchvision import transforms as Transforms
from tqdm import tqdm


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

    return AugmentedTensorDataset(data, labels, transform)

class DefaultKineticsTransform(torch.nn.Module):
    def __init__(self,
                 height:int = 180,
                 ratio:Fraction = Fraction(10,7)
    ):
        super().__init__()
        self.h = height
        self.w = int(ratio * self.h)
        self.transform = Transforms.Resize((self.h, self.w), interpolation=Transforms.InterpolationMode.NEAREST)

    def forward(self, x): # x is the video component of a clip (tensor T, C, H, W)
        clip = x / 255.0
        t, c, h, w = (clip.shape[0], 3, self.h, self.w)
        frames = torch.empty((t, c, h, w))
        for j in range(clip.shape[0]):
            frames[j] = self.transform(clip[j,:,:,:]) # C, H, W
        return torch.transpose(frames, 0, 1) # C, T, H, W



from pytorchvideo.data.encoded_video_pyav import EncodedVideo
from torchvision.transforms import Compose, Lambda
from cabrnet.core.utils.utils_3d import UniformTemporalSubsample

class VideoClip:
    def __init__(self,
             path: str,
             start_sec: Fraction = 0,
             duration_sec: Fraction = 1,
             num_frames: int = 5,
             ratio: Fraction = Fraction(10,7),
             height: int = 180
    ):
        self.clip = load_video_sample(path, start_sec, duration_sec, num_frames, ratio, height)
        self.duration = duration_sec
        self.frames = num_frames
        self.height = height
        self.width = int(ratio*height)

    def toTensor(self):
        return self.clip

def load_video_sample(
        path: str,
        start_sec: int | Fraction = 0,
        duration_sec: int | Fraction = 1,
        num_frames: int = 5,
        ratio: Fraction = Fraction(10, 7),
        height: int = 180
):
    video = EncodedVideo.from_path(path)

    h = height
    w = int(ratio * h)

    start_sec = Fraction(start_sec)
    end_sec = Fraction(start_sec + duration_sec)
    video_data = video.get_clip(start_sec, end_sec)["video"] # C,T,H,W
    transform = Compose([
        UniformTemporalSubsample(num_frames),
        Lambda(lambda x: x/255.0),
        Transforms.Resize((h,w), interpolation=Transforms.InterpolationMode.NEAREST)
    ])

    return transform(video_data)

class KineticsDataset(Dataset):
    def __init__(self,
                 path: str,
                 split: str,
                 subsampling: int,
                 transform: str,
                 clip_length: int,
                 step_between_clips: int,
                 ratio: Fraction,
                 height: int
    ):
        if transform == "default":
            transform = DefaultKineticsTransform(height, ratio)
        else:
            raise ValueError(f"load_kinetics400: unknown transform name \"{transform}\"")

        logger.info(f"Loading Kinetics 400 split {split}...")
        k = Kinetics(path,
                     frames_per_clip= subsampling * clip_length,
                     step_between_clips=step_between_clips,
                     split=split,
                     output_format='TCHW',
                     transform=transform,
                     num_workers=6)

        self.subsampling = subsampling
        self.k = k # samples have shape CTHW

    def __len__(self):
        return len(self.k)

    def __getitem__(self, item):
        sample = self.k[item][0][:,::self.subsampling,:,:] # slice syntax
        return sample, self.k[item][2]

def load_kinetics400 (
        path: str,
        split: str,
        subsampling: int = 2,
        transform: str = "default",
        clip_length: int = 5,
        step_between_clips: int = 200,
        ratio: Fraction = Fraction(10, 7),
        height: int = 180
):
    return KineticsDataset(path, split, subsampling, transform, clip_length, step_between_clips, ratio, height)

from multiprocessing import Pool
class Loader(object):
    def __init__(self, **kwargs):
        self.args = kwargs

    def __call__(self, path):
        return load_video_sample(path,
                                 self.args["start_sec"],
                                 self.args["duration_sec"],
                                 self.args["num_frames"],
                                 self.args["ratio"],
                                 self.args["height"])

class KineticsDatasetNew(Dataset):
    def __init__(self,
                 path: str,
                 split: str,
                 num_frames: int,
                 duration_sec: Fraction,
                 time_between_clips: int,
                 ratio: Fraction,
                 height: int
                 ):
        basedir = os.path.join(path,split)
        subdirs = [ directory for directory in os.scandir(basedir) if os.path.isdir(directory.path) ]
        self.set : list[tuple[torch.Tensor,int]] = []

        with Pool(3) as pool:
            for i,subdir in enumerate(subdirs):
                files = [ file.path for file in os.scandir(subdir) if os.path.isfile(file.path) ]
                logger.info(f"Loading {split}:{subdir.name}...")
                tensors = pool.map(
                    Loader(start_sec=0, duration_sec=duration_sec, num_frames=num_frames, ratio=ratio, height=height),
                    files
                )
                self.set.extend([(t,i) for t in tensors])

    def __len__(self):
        return len(self.set)

    def __getitem__(self, item):
        return self.set[item]

def load_kinetics_new (
        path: str,
        split: str,
        num_frames: int,
        duration_sec: Fraction,
        time_between_clips: int = 5,
        ratio: Fraction = Fraction(10,7),
        height: int = 180
):
    return KineticsDatasetNew(path, split, num_frames, duration_sec, time_between_clips, ratio, height)