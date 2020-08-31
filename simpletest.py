import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, ReLU, MaxPool2D, UpSampling2D

from u2net import U2NET

default_shape = (320, 320, 3)

img = cv2.imread('skateboard.jpg')
img = cv2.resize(img, dsize=default_shape[:2], interpolation=cv2.INTER_CUBIC)

inputs = keras.Input(shape=default_shape)
u2net = U2NET()
outputs = u2net(inputs)
model = keras.Model(inputs=inputs, outputs=outputs, name='u2netmodel')

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=None)
model.summary()
inp = np.expand_dims(img, axis=0)
d1, d2, d3, d4, d5, d6, d7 = model.predict(inp)

pred = d1[0,:,:,:]

fig=plt.figure(figsize=(1, 2))
fig.add_subplot(1, 2, 1)
plt.imshow(inp[0], cmap='gray')
fig.add_subplot(1, 2, 2)
plt.imshow(pred, cmap='gray')
plt.show()