import numpy as np
import torch
from PIL import Image


def get_concat_h_multi_resize(im_list: list[Image.Image], resample=Image.Resampling.BICUBIC):
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

def get_concat_v_multi_resize(im_list: list[Image.Image], resample=Image.Resampling.BICUBIC):
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

def get_concat_tile_resize(im_list_2d: list[list[Image.Image]], resample=Image.Resampling.BICUBIC):
    im_list_v = [get_concat_h_multi_resize(im_list_h, resample=resample) for im_list_h in im_list_2d]
    return get_concat_v_multi_resize(im_list_v, resample=resample)

def to_img(tensor:torch.Tensor):
    nparray = (tensor.numpy()*255).astype(np.uint8)
    return Image.fromarray(nparray)

def chunks(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

path = "./data/prostate/"

def mosaic(dataset:str, slice:int):
    path_ = path + dataset + ".data.pt"
    data = torch.load(path_)
    slices = [ data[i,slice,:,:] for i in range(data.shape[0]) ]
    imgs = [ to_img(x) for x in slices ]
    imgs = list(chunks(imgs, int(np.floor(np.sqrt(len(imgs))))))
    return get_concat_tile_resize(imgs)

def mosaic_gif(dataset:str, duration:int):
    path_ = path + dataset + ".data.pt"
    data = torch.load(path_)
    thickness = data.shape[1]
    frames = [ mosaic(dataset, i ) for i in range(thickness) ]
    frames[0].save(dataset+".gif", save_all=True, append_images=frames[1:], optimize=False, duration=duration, loop=0)

if __name__ == "__main__":
    mosaic_gif("test", 60)