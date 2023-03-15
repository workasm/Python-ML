
from common import *

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as tfkl

def conv_block(ip, nb_filter, dropout_rate=None, weight_decay=1E-4):
    #  x = tfkl.BatchNormalization(mode=0, axis=concat_axis,...
    x = tfkl.Activation('relu')(ip)
    x = tfkl.Conv2D(nb_filter, kernel_size=(3, 3), kernel_initializer="he_uniform", padding="same", use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    if dropout_rate:
        x = tfkl.Dropout(dropout_rate)(x)
    return x

def transition_block(ip, nb_filter, dropout_rate=None, weight_decay=1E-4):

    concat_axis = -1 if tf.keras.backend.image_data_format() == "channels_last" else 1

    x = tfkl.Conv2D(nb_filter, kernel_size=(1, 1), kernel_initializer="he_uniform", padding="same", use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(ip)
    if dropout_rate:
        x = tfkl.Dropout(dropout_rate)(x)
    x = tfkl.AveragePooling2D((2, 2), strides=(2, 2))(x)

    x = tfkl.BatchNormalization(axis=concat_axis, gamma_regularizer=tf.keras.regularizers.l2(weight_decay),
                            beta_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    return x

class DenseBlock(tf.keras.Sequential):
    def __init__(self, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):
        super(DenseBlock, self).__init__()
        for _ in range(nb_layers):
            super(DenseBlock, self).add(tfkl.Conv2D(nb_filter, kernel_size=(1, 1), kernel_initializer="he_uniform", padding="same", use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))

    def forward(self, X):
        for blk in self.layers:
            Y = blk(X)
            X = tfkl.Concatenate(axis=-1)([X, Y])
        return X

def dense_block(input_x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):

    # "th" means (3, 299, 299) - channels first
    # "tf" means (299, 299, 3) - channels last
    #concat_axis = 1 if K.image_dim_ordering() == "th" else -1
    concat_axis = -1 if tf.keras.backend.image_data_format() == "channels_last" else 1

    concat_x = input_x
    for i in range(nb_layers):
        x = conv_block(concat_x, nb_filter, dropout_rate, weight_decay)
        concat_x = tfkl.Concatenate(axis=concat_axis)([concat_x, x])
        nb_filter += growth_rate

    # x1 = input_x
    # x2 = conv_block(x1, nb_filter, dropout_rate, weight_decay)
    # x3 = tfkl.Concatenate(axis=concat_axis)([x1, x2])
    #
    # nb_filter += growth_rate
    # x4 = conv_block(x3, nb_filter, dropout_rate, weight_decay)
    # x5 = tfkl.Concatenate(axis=concat_axis)([x3, x4])
    #
    # nb_filter += growth_rate
    # x6 = conv_block(x5, nb_filter, dropout_rate, weight_decay)
    # x7 = tfkl.Concatenate(axis=concat_axis)([x5, x6])

    return concat_x, nb_filter

def create(nb_classes, img_dim, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=16, dropout_rate=None,
                     weight_decay=1E-4, verbose=True):
    ''' Build the create_dense_net model
    Args:
        nb_classes: number of classes
        img_dim: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        depth: number or layers
        nb_dense_block: number of dense blocks to add to end
        growth_rate: number of filters to add
        nb_filter: number of filters
        dropout_rate: dropout rate
        weight_decay: weight decay
    Returns: keras tensor with nb_layers of conv_block appended
    '''

    model_input = tfkl.Input(shape=img_dim)
    concat_axis = -1 if tf.keras.backend.image_data_format() == "channels_last" else 1

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    # layers in each dense block
    nb_layers = int((depth - 4) / 3)

    # Initial convolution
    x = tfkl.Conv2D(nb_filter, kernel_size=(3, 3), kernel_initializer="he_uniform", padding="same",
                      name="initial_conv2D", use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(model_input)

    x = tfkl.BatchNormalization(axis=concat_axis, gamma_regularizer=tf.keras.regularizers.l2(weight_decay),
                            beta_regularizer=tf.keras.regularizers.l2(weight_decay))(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)
        # add transition_block
        x = transition_block(x, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay)

    # The last dense_block does not have a transition_block
    x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                               weight_decay=weight_decay)

    x = tfkl.Activation('relu')(x)
    x = tfkl.GlobalAveragePooling2D()(x)
    act = 'sigmoid' if nb_classes == 1 else 'softmax'

    x = tfkl.Dense(nb_classes, activation=act,
                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                   bias_regularizer=tf.keras.regularizers.l2(weight_decay))(x)

    densenet = tf.keras.Model(inputs=model_input, outputs=x, name="create_dense_net")
    return densenet