import os, sys, math
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import layers as tfl

def vgg_block(X, num_filters, num_convs, kernel_size=3, useBN=False):
    for _ in range(num_convs):
        X = tfl.Conv2D(num_filters, kernel_size, padding='same')(X)
        if useBN:
            X = tfl.BatchNormalization()(X)
        X = tfl.Activation('relu')(X)
    X = tfl.MaxPool2D((2,2), strides=(2,2))(X)
    return X

# https://github.com/Fjaviervera/DeepEye/blob/master/deepeye.py
def conv_stride_block(X, num_filters, useBN=False):
    X_orig = X
    X = tfl.Conv2D(num_filters, kernel_size=3, padding='same', strides=(2,2))(X)
    if useBN:
        X = tfl.BatchNormalization()(X)

    X = tfl.Activation('relu')(X)
    
    X = tfl.Conv2D(num_filters, kernel_size=3, padding='same')(X)
    if useBN:
        X = tfl.BatchNormalization()(X)

    # tf.keras.layers.AveragePooling2D
    X_orig = tfl.AveragePooling2D(pool_size=(2,2), strides=(2,2),padding='same')(X_orig)
    X_orig = tfl.Conv2D(num_filters, kernel_size=1, padding='same')(X_orig) # increase the number of filters

    X = tfl.Activation('relu')(X + X_orig)
    return X

def conv_dilate_block(X, num_filters, dilate=1, useBN=False):
    X_orig = X
    X = tfl.Conv2D(num_filters, kernel_size=3, padding='same', dilation_rate=dilate)(X)
    if useBN:
        X = tfl.BatchNormalization()(X)

    X = tfl.Activation('relu')(X)

    X = tfl.Conv2D(num_filters, kernel_size=3, padding='same', dilation_rate=dilate)(X)
    if useBN:
        X = tfl.BatchNormalization()(X)

    # this is needed to make sure we have the same number of filters
    #X_orig = tfl.Conv2D(num_filters, kernel_size=1, padding='same')(X_orig)
    X = tfl.Activation('relu')(X + X_orig)
    return X

def conv_block(X, num_filters, useBN=False):
    return conv_dilate_block(X, num_filters, dilate=1, useBN=useBN)

class Augment(tf.keras.layers.Layer):
  def __init__(self, seed=42):
    super().__init__()
    # both use the same seed, so they'll make the same random changes.
    self.model = tf.keras.Sequential([
        tfl.RandomRotation(0.5, seed=seed), #input_shape=input_shape),
        tfl.RandomZoom(height_factor=(-0.2, 0.2), seed=seed),
        tfl.RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode='nearest', seed=seed),
        tfl.RandomContrast(factor=(0.5, 0.5), seed=seed),
    ])
    
  def call(self, inputs): #, labels):
    inputs = self.model(inputs)
    #labels = self.model(labels)
    return inputs


# https://medium.com/analytics-vidhya/different-iou-losses-for-faster-and-accurate-object-detection-3345781e0bf
# https://arxiv.org/pdf/1911.08287v1.pdf
# combination of classification and localization losses: https://arxiv.org/ftp/arxiv/papers/2011/2011.05523.pdf
# shape is: [batch_size, 4]
def DIoU_Loss(y_true_bbox, y_pred_bbox):
     # box: x0, y0, x1, y1
     X0A,Y0A,X1A,Y1A = y_true_bbox
     X0B,Y0B,X1B,Y1B = y_pred_bbox
     zero = tf.constant(0, X0A.dtype)
     areaX = K.minimum(X1A - X0B, X1B - X0A)
     areaX = K.maximum(areaX, zero) # this reduces to 0 if no intersection in X-dir
     areaY = K.minimum(Y1A - Y0B, Y1B - Y0A)
     areaY = K.maximum(areaY, zero) # this reduces to 0 if no intersection in Y-dir
     intersect = areaX*areaY
     union = (X1A - X0A)*(Y1A - Y0A) + (X1B - X0B)*(Y1B - Y0B)
     iou = tf.math.divide_no_nan(intersect, union - intersect)
     minX = K.minimum(X0A, X0B)
     maxX = K.maximum(X1A, X1B)
     minY = K.minimum(Y0A, Y0B)
     maxY = K.maximum(Y1A, Y1B)
     encloseW = K.maximum(zero, maxX - minX)
     encloseH = K.maximum(zero, maxY - minY)
     # squared diagonal of the minimal enclosing box
     diagsq = K.square(encloseW) + K.square(encloseH)

     cAx,cAy = (X0A + X1A)*0.5,(Y0A + Y1A)*0.5
     cBx,cBy = (X0B + X1B)*0.5,(Y0B + Y1B)*0.5
     # squared distance between the centers
     centerDsq = K.square(cAx - cBx) + K.square(cAy - cBy)
     loss = 1.0 - iou + tf.math.divide_no_nan(centerDsq, diagsq)
     return loss, iou ## NOTE: the loss is not reduced!!!

def circle2Bbox(t):
    Q = 1
    X, Y, R = tf.unstack(t, 3, axis=-1)
    return Q, (X - R, Y - R, X + R, Y + R)

class CircleIoU_Metric(tf.keras.metrics.Metric):
    def __init__(self, batchSz, name = 'IoU', **kwargs):
        super(CircleIoU_Metric, self).__init__(**kwargs)
        self.batchSz = tf.cast(batchSz, 'float32')
        self.iou = self.add_weight('iou', initializer = 'zeros')
        self.num = self.add_weight('iou', initializer = 'zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        QA, true_bbox = circle2Bbox(y_true)
        QB, pred_bbox = circle2Bbox(y_pred)
        _, IoU = DIoU_Loss(true_bbox, pred_bbox)
          
        self.iou.assign_add(tf.reduce_sum(IoU))
        self.num.assign_add(self.batchSz)
    def reset_state(self):
        self.iou.assign(0)
        self.num.assign(0)

    def result(self):
        #tf.print(f"Iou = {self.iou} / {self.num}")
        return tf.math.divide_no_nan(self.iou, self.num)


class CicleDIoU_CELoss(tf.keras.losses.Loss):

     def __init__(self, alpha = 0.5):
          super().__init__()
          self.alpha = alpha
          self.CE = tf.keras.losses.BinaryCrossentropy()

     # shape is: [batch_size, 4] qua, x, y, r
     def call(self, y_true, y_pred):
          
          QA, true_bbox = circle2Bbox(y_true)
          QB, pred_bbox = circle2Bbox(y_pred)
          boxLoss, _ = DIoU_Loss(true_bbox, pred_bbox)
          #CEloss = self.CE(QA, QB)
          # (1 - QA)*CEloss + QA*(CEloss + self.alpha * boxLoss)
          # if QA close to zero => bbox does not matter much
          return tf.reduce_min(boxLoss)  #CEloss + QA*(self.alpha * boxLoss))
