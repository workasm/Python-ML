import os, sys, math
import tensorflow as tf
from keras import backend as K
from keras import layers as tfl
   
class UNet:
    def __init__(self):
        self.bn_fn = tfl.BatchNormalization() # axis=-1: normalize channels dimension
        self.activate_fn = tfl.Activation('relu') 

    def encodeBlock(self, input, kernel_size, num_filters, num_blocks, strides=1):

        X = input
        for i in range(num_blocks + 1):
            X = tfl.Conv2D(num_filters, kernel_size, strides=strides, padding='same')(X)
            X = tfl.BatchNormalization()(X) 
            X = tfl.Activation('relu')(X)
    
        X_down = tfl.Conv2D(num_filters*2, (2,2), strides=(2,2), padding='valid')(X)
        X_down = tfl.BatchNormalization()(X_down)
        X_down = tfl.Activation('relu')(X_down)
        return X, X_down

    def decodeBlock(self, X, X_jump, kernel_size, num_filters, num_blocks, strides=1, upsampling = True):
        if X_jump is not None:
            X = tfl.Concatenate()([X, X_jump])
    
        for i in range(num_blocks + 1):
            X = tfl.Conv2D(num_filters, kernel_size, strides=strides, padding='same')(X)
            X = tfl.BatchNormalization()(X) 
            X = tfl.Activation('relu')(X)
        
        if upsampling:
            X = tfl.Conv2DTranspose(num_filters, (2,2), strides=(2,2), padding='valid')(X)
            X = tfl.BatchNormalization()(X) 
            X = tfl.Activation('relu')(X)
        return X

    def __call__(self, inputs):

        X = inputs
        num_filters = 16
        X_stage1, X = self.encodeBlock(X, 3, num_filters, num_blocks=1)

        X_stage2, X = self.encodeBlock(X, 3, num_filters*2, num_blocks=1)

        X_stage3, X = self.encodeBlock(X, 3, num_filters*4, num_blocks=1)

        X_stage4, X = self.encodeBlock(X, 3, num_filters*8, num_blocks=1)

        X = self.decodeBlock(X, None, 3, num_filters*16, num_blocks=1)

        X = self.decodeBlock(X, X_stage4, 3, num_filters*16, num_blocks=1)

        X = self.decodeBlock(X, X_stage3, 3, num_filters*8, num_blocks=1)

        X = self.decodeBlock(X, X_stage2, 3, num_filters*4, num_blocks=1)

        X = self.decodeBlock(X, X_stage1, 3, num_filters*2, num_blocks=1, strides=1, upsampling=False)

        X = tfl.Conv2D(filters = 3, kernel_size = (1,1), padding = 'valid')(X)
        X = tfl.Activation("softmax")(X)

        return tf.keras.Model(inputs=inputs, outputs=X)


    