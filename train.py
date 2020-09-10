import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pathlib
import signal

from config import *
from dataloader import *
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, ReLU, MaxPool2D, UpSampling2D
from model.u2net import U2NET


# Arguments
parser = argparse.ArgumentParser(description='U^2-NET Salient Object Detection')
parser.add_argument('--batch_size', default=None, type=int,
                    help='Training batch size')
parser.add_argument('--lr', '--learning_rate', default=None, type=float,
                    help='Optimizer learning rate. Default: %s' % learning_rate)
parser.add_argument('--save_interval', default=None, type=int,
                    help='How many iterations between saving of model weights')
parser.add_argument('--model_file', default=None, type=str,
                    help='Output location for model weights. Default: %s' % model_file)
parser.add_argument('--resume', default=None, type=str,
                    help="Resume training network from saved weights file. Leave as None to start new training.")
args = parser.parse_args()

if args.batch_size:
    batch_size = args.batch_size

if args.lr:
    learning_rate = args.lr

if args.save_interval:
    save_interval = args.save_interval

if args.model_file:
    model_file = args.model_file

if args.resume:
    resume = args.resume

# Previewing, not necessary
img = cv2.imread('examples/skateboard.jpg')
img = cv2.resize(img, dsize=default_in_shape[:2], interpolation=cv2.INTER_CUBIC)
inp = np.expand_dims(img, axis=0)

# Optimizer / Loss
adam = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=.9, beta_2=.999, epsilon=1e-08)
bce = keras.losses.BinaryCrossentropy()
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='model.ckpt', save_weights_only=True, verbose=1)

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

    if resume:
        print('Loading weights from %s' % resume)
        model.load_weights(resume)

    # setup the dataset if the user hasn't set it up yet
    download_duts_tr_dataset()

    # helper function to save state of model
    def save_weights():
        print('Saving state of model to %s' % model_file)
        model.save_weights(str(model_file))
    
    # signal handler for early abortion to autosave model state
    def autosave(sig, frame):
        print('Training aborted early... Saving weights.')
        save_weights()
        exit(0)

    for sig in [signal.SIGABRT, signal.SIGINT, signal.SIGTSTP]:
        signal.signal(sig, autosave)

    # start training
    print('Starting training')
    for e in range(epochs):
        feed, out = load_training_batch(batch_size=batch_size)
        loss = model.train_on_batch(feed, out)

        if e % 10 == 0:
            print('[%05d] Loss: %.4f' % (e, loss))

        if save_interval and e > 0 and e % save_interval == 0:
            save_weights()

if __name__=="__main__":
    train()