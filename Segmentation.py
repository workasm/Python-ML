import os, sys, math
import tensorflow as tf
from keras import backend as K
from keras import layers as tfl

# this just adds activation and batch norm to Conv2D layer
class ConvBlock(tfl.Conv2D):
    def __init__(self, nonlinear, **kwargs):
        super().__init__(**kwargs)
        self.activation_fn = tfl.Activation(nonlinear)
        self.bnorm_fn = tfl.BatchNormalization() 
   
    def call(self, X):
        X = super().call(X)
        X = self.bnorm_fn(X) 
        X = self.activation_fn(X)
        return X

class UNet:
    def __init__(self):
        self.bn_fn = tfl.BatchNormalization() # axis=-1: normalize channels dimension
        self.activate_fn = tfl.ReLU()

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

def loss_initializer(self):
        labels_linear = tf.reshape(tensor=self.labels, shape=[-1])
        not_ignore_mask = tf.to_float(tf.not_equal(labels_linear, self.ignore_label))
        # The locations represented by indices in indices take value on_value, while all other locations take value off_value.
        # For example, ignore label 255 in VOC2012 dataset will be set to zero vector in onehot encoding (looks like the not ignore mask is not required)
        onehot_labels = tf.one_hot(indices=labels_linear, depth=self.num_classes, on_value=1.0, off_value=0.0)

        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=tf.reshape(self.outputs, shape=[-1, self.num_classes]), weights=not_ignore_mask)

        return loss

# Atrous Spatial Pyramid Pooling (ASPP) Block
def ASPP_block(inputs, baseModel, num_filters, dilation_rates, relu_fn='relu'):

    features = baseModel(inputs)
    X = features
    shape = X.shape # tuple of 4 params
    use_bias=False

    # TODO: check this with non-square inputs !!
    #outMean = tfl.AveragePooling2D(pool_size=(shape[-3], shape[-2]))(X)  
    outMean = tfl.GlobalAveragePooling2D(keepdims=True)(X)
    outMean = ConvBlock(relu_fn, filters=num_filters, kernel_size=1, padding='same', use_bias=True)(outMean)
    newSz = (shape[-3]//outMean.shape[1], shape[-2]//outMean.shape[2])
    outMean = tfl.UpSampling2D(size=newSz, interpolation='bilinear')(outMean)
    #outMean = tf.reduce_mean(X, axis=[1,2], keepdims=True)
    #outMean = tf.image.resize(outMean, size=[shape[1],shape[2]], method=tf.image.ResizeMethod.BILINEAR)

    out1 = ConvBlock(relu_fn, filters=num_filters, kernel_size=1, padding='same', use_bias=use_bias)(X)
    dilated = [outMean, out1]
    for rate in dilation_rates:
        outX = ConvBlock(relu_fn, filters=num_filters, kernel_size=3, padding='same', dilation_rate=rate, use_bias=use_bias)(X)
        dilated.append(outX)

    X = tfl.Concatenate(axis=-1)(dilated)
    X = ConvBlock(relu_fn, filters=num_filters, kernel_size=1, padding='same', use_bias=use_bias)(X)
    #X = tfl.Activation()

    return tf.keras.Model(inputs=inputs, outputs=X)
