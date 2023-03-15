import os, sys, math
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as tfl
import tensorflow.keras.backend as K

class CAE_generic:
    def __init__(self,bottlenck_size,if_batch_norm=False,
                 if_extra_dense=False,**kwargs):
        self.bottlenck_size = bottlenck_size
        self.if_batch_norm = if_batch_norm
        self.if_extra_dense = if_extra_dense

    def complete(self, input_size):
        inputs = tfl.Input(shape=input_size)
        X = self.encoder_fun(inputs)
        X = self.decoder_fun(X)

        return keras.Model(inputs=inputs, outputs=X)
       
    def encoder_fun(self, input):
        vgg_layer = self.vgg_block(input,32,2)
        vgg_layer = self.vgg_block(vgg_layer,64,2)
        vgg_layer = self.vgg_block(vgg_layer,128,2)
        flattening = tfl.Flatten()(vgg_layer)
        if self.if_extra_dense:
            extra_dense1 = tfl.Dense(4*self.bottlenck_size,activation="relu",name="extra_dense1")(flattening)
            extra_dense2 = tfl.Dense(2*self.bottlenck_size,activation="relu",name="extra_dense2")(extra_dense1)
            
            out = tfl.Dense(self.bottlenck_size,activation="relu")(extra_dense2)
            self.units = flattening.shape[1]
            self.last_con_shape = vgg_layer.shape
        else:
            out = tfl.Dense(self.bottlenck_size,activation="relu")(flattening)
            self.units = flattening.shape[1]
            self.last_con_shape = vgg_layer.shape
        #encoder = keras.Model(inputs=input_,outputs=out,name="encoder")
        #return encoder
        return out
    
    def decoder_fun(self, input):
        #input_ = tfl.Input(shape = self.bottlenck_size)
        x = tfl.Dense(self.units,activation="relu")(input)
        x = tfl.Reshape(self.last_con_shape[1:])(x)
        x = tfl.Conv2DTranspose(128,3,strides=2,padding="same")(x)
        if self.if_batch_norm:
            x = tfl.BatchNormalization()(x)
        x = tfl.Activation('relu')(x)
        x = tfl.Conv2DTranspose(64,3,strides=2,padding="same")(x)
        if self.if_batch_norm:
            x = tfl.BatchNormalization()(x)
        x = tfl.Activation('relu')(x)
        x = tfl.Conv2DTranspose(32,3,strides=2,padding="same")(x)
        if self.if_batch_norm:
            x = tfl.BatchNormalization()(x)
        x = tfl.Activation('relu')(x)
        x = tfl.Conv2DTranspose(1,3,strides=1,padding="same")(x)
        output = tfl.Activation('sigmoid')(x)
        return output
        #decoder = keras.Model(inputs=input_,outputs=output,name="decoder")
        #return decoder
    
    def vgg_block(self, X, num_filters, num_convs, kernel_size=3):
        for _ in range(num_convs):
            X = tfl.Conv2D(num_filters, kernel_size, padding='same')(X)
            if self.if_batch_norm:
                X = tfl.BatchNormalization()(X)
            X = tfl.Activation('relu')(X)
        X = tfl.MaxPool2D((2,2), strides=(2,2))(X)
        return X

class Augment(tf.keras.layers.Layer):
  def __init__(self, seed=42):
    super().__init__()
    # both use the same seed, so they'll make the same random changes.
    self.model = keras.Sequential([
        tfl.RandomRotation(0.5, seed=seed), #input_shape=input_shape),
        tfl.RandomZoom(height_factor=(-0.2, 0.2), seed=seed),
        tfl.RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode='nearest', seed=seed),
        tfl.RandomContrast(factor=(0.5, 0.5), seed=seed),
    ])
    
  def call(self, inputs): #, labels):
    inputs = self.model(inputs)
    #labels = self.model(labels)
    return inputs