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

def str2bool(v):
    return v is not None and v.lower() in ("yes", "true", "t", "1")

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
parser.add_argument('--merged', default=False, type=str2bool,
                    help='Display image and output mask side-by-side in each output image')
parser.add_argument('--apply_mask', default=False, type=str2bool,
                    help='Apply the mask to the input image and save the product.')
args = parser.parse_args()

if args.output:
    output_dir = pathlib.Path(args.output)

def apply_mask(img, mask):
    return np.multiply(img[0], np.tile(mask[0][0], [1, 1, 3]))

def main():
    input_images = []

    if args.image:
        assert os.path.exists(args.image)
        input_images.append(args.image)
    
    if args.images:
        input_dir = pathlib.Path(args.images)
        if not input_dir.exists():
            input_dir.mkdir()
        imgs = glob.glob(str(input_dir.joinpath('*png'))) + glob.glob(str(input_dir.joinpath('*.jpg')))
        assert len(imgs) > 0, 'No images found in directory %s' % str(input_dir)
        input_images.extend(imgs)
        
    if not output_dir.exists():
        output_dir.mkdir()
    
    if len(input_images) == 0:
        return
    
    inputs = keras.Input(shape=default_in_shape)
    net = U2NET()
    out = net(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name='u2netmodel')
    model.compile(optimizer=adam, loss=bce_loss, metrics=None)

    if args.weights:
        assert(os.path.exists(args.weights))
        model.load_weights(args.weights)

    for img in input_images:
        input_image, size = load_test_image(img)
        mask = model(input_image)

        plt.margins(0, 0)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        if args.merged:
            f, a = plt.subplots(1,2)
            f.subplots_adjust(hspace=0, wspace=0)
            a[0].set_axis_off()
            a[1].set_axis_off()

            out_mask = np.tile(mask[0][0], [1, 1, 3])
            a[0].imshow(out_mask, cmap='gray', vmin=0, vmax=1)
            a[1].imshow(apply_mask(input_image, mask) if args.apply_mask else input_image[0])
        else:
            f, a = plt.gcf(), plt.gca()
            a.set_axis_off()
            f.subplots_adjust(hspace=0, wspace=0)
            if args.apply_mask:
                a.imshow(apply_mask(input_image, mask))
            else:
                a.imshow(np.tile(mask[0][0], [1, 1, 3]), cmap='gray', vmin=0, vmax=1)

        o = output_dir.joinpath(pathlib.Path(img).name)
        f.savefig(o, transparent=True, bbox_inches='tight', pad_inches=0)
        
if __name__=='__main__':
    main()