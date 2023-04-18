
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from IPython.display import display, Image
#import pandas as pd    

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
#tf.debugging.set_log_device_placement(True)

def read_bcrf(filepath):
    with open(filepath, 'rb') as f:
        header = f.read(4096).decode('utf-8', 'ignore').split('\n')
        data = f.read()
        
    output = {}
    xcvt,ycvt,zcvt = 1,1,1
    for item in header:
        item = item.replace('\x00', '')
        val = item[item.find('=') + 1:]

        if item.find('xpixels') != -1:
            output['x-pixels'] = int(val)
        elif item.find('ypixels') != -1:
            output['y-pixels'] = int(val)
        elif item.find('xlength') != -1:
            output['x-length'] = float(val)
        elif item.find('ylength') != -1:
            output['y-length'] = float(val)
        elif item.find('zunit') != -1:
            if 'nm' in item:
                zcvt = 1000
            elif 'um' in item:
                zcvt = 1
            else:
                zcvt = 1
        elif item.find('xunit') != -1:
            if 'nm' in item:
                xcvt = 1000000
            elif 'um' in item:
                xcvt = 1000
        elif item.find('yunit') != -1:
            if 'nm' in item:
                ycvt = 1000000
            elif 'um' in item:
                ycvt = 1000
    
    #convert length data to mm
    output['x-length'] = output['x-length'] / xcvt
    output['y-length'] = output['y-length'] / ycvt

    num = len(data)//4
    fdata = np.frombuffer(data, dtype=np.float32, count=num)
    fdata = np.reshape(fdata, 
                    (output['y-pixels'], output['x-pixels'])) / zcvt
    #replace void markers with nan values
    output['Data'] = np.where(fdata == (3.4028234663852886e+38/zcvt), 
                    np.nan, fdata) #3...e38 represents void in bcrf
    output['filename'] = str(os.path.basename(filepath))
    return output

class Normalize(tf.Module):
  def __init__(self, x):
    # Initialize the mean and standard deviation for normalization
    self.mean = tf.Variable(tf.math.reduce_mean(x))
    self.std = tf.Variable(tf.math.reduce_std(x))

  def norm(self, x):
    # Normalize the input
    return (x - self.mean)/self.std

  def unnorm(self, x):
    # Unnormalize the input
    return (x * self.std) + self.mean

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

def displayModel(M):
    tf.keras.utils.plot_model(M, to_file='model.png', show_shapes=True, 
          show_layer_activations=True, show_layer_names=True)
    display(Image('model.png'))

def display3Dtensor(images3D, dims=[5,5]):
    images_arr = tf.split(images3D, images3D.shape[0], axis=0)
    displayImages(images_arr, dims)

def displayImages(images_arr, dims=[5,5]):
    fig, axes = plt.subplots(dims[0], dims[1], figsize=(10,10))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        # squeeze unnecessary unit dimensions
        img2 = tf.squeeze(img)
        ax.imshow(img2, cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def displayDS(ds, num, func = lambda x: tf.keras.utils.array_to_img(x), figsize=(10, 10), columns=3):
    rows = (num + columns - 1)//columns
    _, axs = plt.subplots(rows, columns, figsize=(12, 12), squeeze=False)
    axs = axs.flatten()
    for i, item in zip(range(num), ds):
        img = func(*item)
        axs[i].imshow(img, cmap="gray")
        axs[i].axis('off')

    for i in range(num, len(axs)):    
        axs[i].set_visible(False)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

def drawCircleImg(img, bbox, frameName = '', color='#ff00ff'):
    X, Y, R = bbox
    if tf.rank(img) != 3:
        img = tf.expand_dims(img, axis=-1)
    II = tf.keras.utils.array_to_img(img)
    tf.print(X,Y,R)
    #if Q == 0:
    #    return II
    II = II.convert(mode='RGB')
    ehc = ImageEnhance.Brightness(II)
    II = ehc.enhance(2)
    draw = ImageDraw.Draw(II)
    shape = img.shape
    X, Y, R = X * shape[0], Y * shape[1], R * shape[0]
    draw.ellipse([X-R,Y-R,X+R,Y+R], outline=color)
    if frameName != '':
        tf.print(frameName)
    return II 

def pandas():    
    # uu = pd.DataFrame()
    # for f in csv_set:
    #     fname = bytes.decode(f.numpy())
    #     zz = pd.read_csv(fname, sep=';', header=1, usecols=[0,1,2,3,4,5], skipinitialspace=True,
    #             names=['FrameNo','EyeCode','PupilQ','PupilX','PupilY','PupilR'],
    #             dtype = {'EyeCode' : str } )
    #     #zz.head()
    #     zz['FileName'] = fname
    #     uu = uu.append(zz)
    #names=['FrameNo','EyeCode','PupilQ','PupilX','PupilY','PupilR','CycloQ','Angle','PSTValue','RadStdDev','AvgGradient','MaxHoughCenter'])
    # uu.head()
    # uu.info()
    # zz = uu.pop('EyeCode')
    # gg = tf.data.Dataset.from_tensor_slices(uu.values)
    # print(next(iter(uu)))
    return