import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, ReLU, MaxPool2D, UpSampling2D

from model.u2net import *
from config import *
from dataloader import *

# Args
parser = argparse.ArgumentParser(description='U^2-NET Testing')
parser.add_argument('--image', default=None, type=str,
                    help='Evaluate the model on a single image')
parser.add_argument('--images', default=None, type=str,
                    help='Evaluate the model on a directory of images (.jpg & .png')
parser.add_argument('--output', default=None, type=str,
                    help='The directory to output to (default=out/)')
parser.add_argument('--weights', default=None, type=str,
                    help='The weights file of a trained network')
args = parser.parse_args()

if args.output:
    output_dir = pathlib.Path(args.output)

def main():
    input_images = []

    if args.image:
        assert(os.path.exists(args.image))
        input_images.append(args.image)
    
    if args.images:
        assert(os.path.exists(args.images))
        input_dir = pathlib.Path(args.images)
        imgs = glob.glob(str(input_dir.joinpath('*png'))) + glob.glob(str(input_dir.joinpath('*.jpg')))
        input_images.extend(imgs)
        
    if not output_dir.exists():
        os.mkdir(str(args.output))
    
    if len(input_images) == 0:
        return
    
    inputs = keras.Input(shape=default_in_shape)
    net    = U2NET()
    out    = net(inputs)
    model  = keras.Model(inputs=inputs, outputs=out, name='u2netmodel')
    model.compile(optimizer=adam, loss=bce_loss, metrics=None)

    if args.weights:
        assert(os.path.exists(args.weights))
        model.load_weights(args.weights)

    for img in input_images:
        single_img, size = load_test_image(img)
        evaluation = model(single_img)
        f, a = plt.subplots(1,2)
        f.subplots_adjust(hspace=0, wspace=0)
        a[0].set_xticks([])
        a[0].set_yticks([])
        a[0].imshow(np.tile(evaluation[0][0], [1, 1, 3]), cmap='gray', vmin=0, vmax=1)
        a[1].imshow(single_img[0])
        a[1].set_xticks([])
        a[1].set_yticks([])

        o = output_dir.joinpath(pathlib.Path(img).name)
        f.savefig(o, transparent=True, bbox_inches='tight')
        
if __name__=='__main__':
    main()