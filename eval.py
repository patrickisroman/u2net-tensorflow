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
                    help='Display input image and output mask side-by-side in each output image')
parser.add_argument('--apply_mask', default=False, type=str2bool,
                    help='Apply the mask to the input image and show in place of the mask')
args = parser.parse_args()

if args.output:
    output_dir = pathlib.Path(args.output)

def apply_mask(img, mask):
    return np.multiply(img, mask)

def main():
    input_images = []

    if args.image:
        assert os.path.exists(args.image), "Input image file must exist: %s" % args.image
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
        assert os.path.exists(args.weights), 'Model weights path must exist: %s' % args.weights
        model.load_weights(args.weights)

    # evaluate each image
    for img in input_images:
        image = Image.open(img).convert('RGB')
        input_image = image
        if image.size != default_in_shape:
            input_image = image.resize(default_in_shape[:2], Image.BICUBIC)
        
        input_tensor = format_input(input_image)
        fused_mask_tensor = model(input_tensor, Image.BICUBIC)[0][0]
        output_mask = np.asarray(fused_mask_tensor)
        
        if image.size != default_in_shape:
            output_mask = cv2.resize(output_mask, dsize=image.size)
        
        output_mask = np.tile(np.expand_dims(output_mask, axis=2), [1, 1, 3])
        output_image = np.expand_dims(np.array(image)/255., 0)[0]
        if args.apply_mask:
            output_image = apply_mask(output_image, output_mask)
        else:
            output_image = output_mask

        if args.merged:
            output_image = np.concatenate((output_mask, output_image), axis=1)

        output_image = cv2.cvtColor(output_image.astype('float32'), cv2.COLOR_BGR2RGB) * 255.
        output_location = output_dir.joinpath(pathlib.Path(img).name)
        cv2.imwrite(str(output_location), output_image)
        
if __name__=='__main__':
    main()