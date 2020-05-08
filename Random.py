#Random static generator using the NN

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(2*2*40, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((2,2,40)))
    assert model.output_shape == (None, 2, 2, 40)
    model.add(layers.Conv2DTranspose(30, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 4, 4, 30)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(20, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 20)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(10, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 10)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(5, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 5)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 64, 1)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    return model

model = make_generator_model()
noise = tf.random.normal([1, 100])
img = model(noise, training=False)
plt.imshow(img[0, :, :, 0], cmap="gray")
plt.show()
