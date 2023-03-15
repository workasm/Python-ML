
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as tfkl
import matplotlib.pyplot as plt

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
#tf.debugging.set_log_device_placement(True)

def decode_raw(file_name):
    IMAGE_W = 640
    IMAGE_H = 480
    bytes = tf.io.read_file(file_name)
    image = tf.io.decode_raw(bytes, tf.uint8)
    image = tf.reshape(image, [IMAGE_H, IMAGE_W])
    return tf.cast(image, tf.float32) * (1. / 255) - 0.5

def decode_png(name):
    bytes = tf.io.read_file(name)
    image = tf.io.decode_png(bytes)
    return tf.image.convert_image_dtype(image, tf.float32)

def plot3Dtensor(images3D, dims=[5,5]):
    images_arr = tf.split(images3D, images3D.shape[0], axis=0)
    plotImages(images_arr, dims)

def plotImages(images_arr, dims=[5,5]):
    fig, axes = plt.subplots(dims[0], dims[1], figsize=(10,10))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        # squeeze unnecessary unit dimensions
        img2 = tf.squeeze(img)
        ax.imshow(img2, cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
