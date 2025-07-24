import torch
import numpy as np
from fractions import Fraction
from PIL import Image

# slice up the volume (C, T, H, W) in T different Images
def frames_from_sample(sample:torch.Tensor) -> list[Image]:
    t = sample.shape[1]
    frames = []

    if sample.dim() == 4:
        frames = [ sample[:, idx, :, :] for idx in range(t) ]  # CTHW -> CHW
    elif sample.dim() == 3:
        frames = [ sample[idx, :,:] for idx in range(t) ]      # THW -> HW
    else:
        assert False, "sample should either be 3-dimensional (THW) or 4-dimensional (CTHW)"

    for i, frame in enumerate(frames):
        if sample.dim() == 4:
            frame = frame.transpose(0,2).transpose(0,1) # CHW -> HWC
        frame = (frame.numpy() * 255).astype(np.uint8)
        img = Image.fromarray(frame)
        frames[i] = img

    return frames

def image_from_3d(slices:list[Image]):
    # arrange the pictures in a (roughly) square grid
    def chunks(lst: list, n: int):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    assert len(slices) > 0, "The image is empty"
    height = int(np.ceil(np.sqrt(len(slices))))
    slices = list(chunks(slices, height))

    # produce a mosaic from the pictures.
    # mosaic functions from:
    # https://note.nkmk.me/en/python-pillow-concat-images/

    def get_concat_h_multi_resize(im_list: list[Image], resample=Image.Resampling.BICUBIC):
        min_height = min(im.height for im in im_list)
        im_list_resize = [im.resize((int(im.width * min_height / im.height), min_height), resample=resample)
                          for im in im_list]
        total_width = sum(im.width for im in im_list_resize)
        dst = Image.new('RGB', (total_width, min_height))
        pos_x = 0
        for im in im_list_resize:
            dst.paste(im, (pos_x, 0))
            pos_x += im.width
        return dst

    def get_concat_v_multi_resize(im_list: list[Image], resample=Image.Resampling.BICUBIC):
        min_width = min(im.width for im in im_list)
        im_list_resize = [im.resize((min_width, int(im.height * min_width / im.width)), resample=resample)
                          for im in im_list]
        total_height = sum(im.height for im in im_list_resize)
        dst = Image.new('RGB', (min_width, total_height))
        pos_y = 0
        for im in im_list_resize:
            dst.paste(im, (0, pos_y))
            pos_y += im.height
        return dst

    def get_concat_tile_resize(im_list_2d: list[list[Image]], resample=Image.Resampling.BICUBIC):
        im_list_v = [get_concat_h_multi_resize(im_list_h, resample=resample) for im_list_h in im_list_2d]
        return get_concat_v_multi_resize(im_list_v, resample=resample)

    return get_concat_tile_resize(slices)

# copied from the pytorchvideo codebase because they're still using deprecated functions
# and python won't import UniformTemporalSubsample
def uniform_temporal_subsample(
    x: torch.Tensor, num_samples: int, temporal_dim: int = -3
) -> torch.Tensor:
    """
    Uniformly subsamples num_samples indices from the temporal dimension of the video.
    When num_samples is larger than the size of temporal dimension of the video, it
    will sample frames based on nearest neighbor interpolation.

    Args:
        x (torch.Tensor): A video tensor with dimension larger than one with torch
            tensor type includes int, long, float, complex, etc.
        num_samples (int): The number of equispaced samples to be selected
        temporal_dim (int): dimension of temporal to perform temporal subsample.

    Returns:
        An x-like Tensor with subsampled temporal dimension.
    """
    t = x.shape[temporal_dim]
    assert num_samples > 0 and t > 0
    # Sample by nearest neighbor interpolation if num_samples > t.
    indices = torch.linspace(0, t - 1, num_samples)
    indices = torch.clamp(indices, 0, t - 1).long()
    return torch.index_select(x, temporal_dim, indices)

class UniformTemporalSubsample(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.uniform_temporal_subsample``.
    """

    def __init__(self, num_samples: int, temporal_dim: int = -3):
        """
        Args:
            num_samples (int): The number of equispaced samples to be selected
            temporal_dim (int): dimension of temporal to perform temporal subsample.
        """
        super().__init__()
        self._num_samples = num_samples
        self._temporal_dim = temporal_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        return uniform_temporal_subsample(
            x, self._num_samples, self._temporal_dim
        )

# Some debug functions...
from cabrnet.core.utils.load_3d import load_video_sample

def convert_to_img(
        video: str | torch.Tensor,
        duration: int | Fraction = 1,
        num_frames: int = 9,
        ratio: int | Fraction = Fraction(10,7),
        height:int = 180
):
    if isinstance(video, str):
        v = load_video_sample(video, 0, duration, num_frames, ratio, height)
    else:
        v = video

    f = frames_from_sample(v)
    return image_from_3d(f)

def save_as_gif(
        path: str,
        output: str = "output.gif",
        duration: int | Fraction = 1,
        num_frames: int = 9,
        ratio: int | Fraction = Fraction(10,7),
        height:int = 180
):
    v = load_video_sample(path, 0, duration, num_frames, ratio, height)
    f = frames_from_sample(v)
    f[0].save(output, save_all=True, append_images=f[1:], optimize=False, duration=duration, loop=0)