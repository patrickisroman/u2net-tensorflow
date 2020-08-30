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
out = model.predict(inp)

plt.imshow(img)
plt.show()