import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pathlib

from dataloader import *

from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, ReLU, MaxPool2D, UpSampling2D

from model.u2net import U2NET

# Model definition
default_in_shape  = (320, 320, 3)
default_out_shape = (320, 320, 1)
batch_size = 12
epochs = 10000

# Previewing, not necessary
img = cv2.imread('examples/skateboard.jpg')
img = cv2.resize(img, dsize=default_in_shape[:2], interpolation=cv2.INTER_CUBIC)
inp = np.expand_dims(img, axis=0)

# Dataset
dataset = 'data/dutstr/'
dataset_path = pathlib.Path(dataset)
image_path = dataset_path.joinpath('image')
mask_path = dataset_path.joinpath('mask')

# Optimizer / Loss
adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=.9, beta_2=.999, epsilon=1e-08)
bce = keras.losses.BinaryCrossentropy()
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='model.ckpt', save_weights_only=True, verbose=1)

# Args
parser = argparse.ArgumentParser(description='U^2-NET Salient Object Detection')
args = parser.parse_args()

def bce_loss(y_true, y_pred):
    y_p = tf.expand_dims(y_pred, axis=-1)
    loss0 = bce(y_true, y_p[0])
    loss1 = bce(y_true, y_p[1])
    loss2 = bce(y_true, y_p[2])
    loss3 = bce(y_true, y_p[3])
    loss4 = bce(y_true, y_p[4])
    loss5 = bce(y_true, y_p[5])
    loss6 = bce(y_true, y_p[6])

    return loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

def normalize(v):
    max_val = tf.reduce_max(v)
    min_val = tf.reduce_min(v)
    dn = (v-min_val)/(max_val-min_val)
    return dn

def train():
    inputs = keras.Input(shape=default_in_shape)
    net    = U2NET()
    out    = net(inputs)
    model  = keras.Model(inputs=inputs, outputs=out, name='u2netmodel')
    model.compile(optimizer=adam, loss=bce_loss, metrics=None)
    model.summary()

    for e in range(epochs):
        feed, out = load_training_batch(batch_size=batch_size)
        loss = model.train_on_batch(feed, out)

if __name__=="__main__":
    train()