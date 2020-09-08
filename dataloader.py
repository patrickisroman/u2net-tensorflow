import os
import pathlib
import random
import numpy as np
from PIL import Image 

default_in_shape = (320, 320, 3)
default_out_shape = (320, 320, 1)

root_data_dir = pathlib.Path('data/datasets/')
dataset_dir = root_data_dir.joinpath('DUTS-TR')
image_dir = dataset_dir.joinpath('DUTS-TR-Image')
mask_dir = dataset_dir.joinpath('DUTS-TR-Mask')

cache = None

def get_image_mask_pair(img_name, in_resize=None, out_resize=None):
    img  = Image.open(image_dir.joinpath(img_name))
    mask = Image.open(mask_dir.joinpath(img_name.replace('jpg', 'png')))

    if in_resize:
        img = img.resize(in_resize[:2], Image.BICUBIC)
    
    if out_resize:
        mask = mask.resize(out_resize[:2], Image.BICUBIC)

    return (np.asarray(img, dtype=np.float32), np.expand_dims(np.asarray(mask, dtype=np.float32), -1))

def load_training_batch(batch_size=8, in_shape=default_in_shape, out_shape=default_out_shape):
    global cache
    if cache is None:
        cache = os.listdir(image_dir)
    
    imgs = random.choices(cache, k=batch_size)
    image_list = [get_image_mask_pair(img, in_resize=default_in_shape, out_resize=default_out_shape) for img in imgs]
    
    tensor_in  = np.stack([i[0] for i in image_list])
    tensor_out = np.stack([i[1] for i in image_list])
    
    return (tensor_in, tensor_out)
