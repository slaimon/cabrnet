import torch
from fractions import Fraction
from cabrnet.core.utils.utils_3d import frames_from_sample, image_from_3d
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