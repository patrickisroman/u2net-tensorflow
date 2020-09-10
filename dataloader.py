import os
import pathlib
import random
import numpy as np
import wget
import zipfile
import glob

from config import *
from PIL import Image 

cache = None

# aborting wget leaves .tmp files everywhere >:(
def clean_dataloader():
    for tmp_file in glob.glob('*.tmp'):
        os.remove(tmp_file)

def download_duts_tr_dataset():
    if dataset_dir.exists():
        return

    print('Downloading DUTS-TR Dataset from %s...' % dataset_url)
    f = wget.download(dataset_url, out=str(root_data_dir.absolute()))
    if not pathlib.Path(f).exists():
        return

    print('Extracting dataset...')
    with zipfile.ZipFile(f, 'r') as zip_file:
        zip_file.extractall(root_data_dir.absolute())

    clean_dataloader()

def get_image_mask_pair(img_name, in_resize=None, out_resize=None):
    img  = Image.open(image_dir.joinpath(img_name))
    mask = Image.open(mask_dir.joinpath(img_name.replace('jpg', 'png')))

    if in_resize:
        img = img.resize(in_resize[:2], Image.BICUBIC)
    
    if out_resize:
        mask = mask.resize(out_resize[:2], Image.BICUBIC)

    return (np.asarray(img, dtype=np.float32), np.expand_dims(np.asarray(mask, dtype=np.float32), -1))

def load_training_batch(batch_size=12, in_shape=default_in_shape, out_shape=default_out_shape):
    global cache
    if cache is None:
        cache = os.listdir(image_dir)
    
    imgs = random.choices(cache, k=batch_size)
    image_list = [get_image_mask_pair(img, in_resize=default_in_shape, out_resize=default_out_shape) for img in imgs]
    
    tensor_in  = np.stack([i[0]/255. for i in image_list])
    tensor_out = np.stack([i[1]/255. for i in image_list])
    
    return (tensor_in, tensor_out)