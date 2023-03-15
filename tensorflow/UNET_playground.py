
from __future__ import absolute_import, division, print_function, unicode_literals

from sympy import nfloat

from common import *
import misc_nets
import math

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib
import numpy as np
#import cv_image_proc
# how to start computation
# runfile('UNET_playground.py', wdir='C:/work/tensorflow')
# note batch size 4 does not work at all!
BATCH_SIZE = 2
EPOCHS = 16

data_dir = pathlib.Path("C:/work/titan/TitanTestSets/FlapCutImages")

# for f,l in image_set.take(25):
#   print(f[0].numpy(),f[1].numpy())

#train_size = int(0.7 * data_size)
#val_size = int(0.15 * data_size)
#test_size = int(0.3 * data_size)
#STEPS_PER_EPOCH = train_size // BATCH_SIZE
#train_batch = next(iter(test_set))
#z1,z2 = tf.split(train_batch[0], 2, -1)
#plotImages(z1.numpy())

def adjustData(img,mask):
    if(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        #img[mask > 0.5] = img[mask > 0.5]*2
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)

def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode ="grayscale",
                  mask_color_mode = "grayscale", image_save_prefix  = "image", mask_save_prefix  = "mask",
                  save_to_dir = None, target_size = (256,256), seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img, mask)
        yield (img,mask)

def testGenerator(batch_size, train_path, image_folder, image_color_mode ="grayscale",
                           target_size = (256,256), seed = 1):
    image_datagen = ImageDataGenerator(rescale=1./255)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        seed = seed)
    for img in image_generator:
        yield img

# what to try: unroll eye over the pupil center
# mark boundaries of the mask on the unrolled image
# optionally we could also do it on PST filtered image

input_size=(256,256)
data_gen_args = dict(rotation_range=15,
                    #width_shift_range=0.05,
                    #height_shift_range=0.05,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest')

myGene = trainGenerator(BATCH_SIZE, data_dir / 'train2', 'images', 'labels',
                        data_gen_args, save_to_dir = None, target_size=input_size)

testGene = testGenerator(1, data_dir / 'test', 'images', target_size=input_size)
#imgs,masks = next(myGene)
#imgs = next(testGene)
dw=math.ceil(math.sqrt(BATCH_SIZE))
dh=math.ceil(BATCH_SIZE/dw)
#plot3Dtensor(imgs, dims=[dw,dh])
#plot3Dtensor(masks, dims=[dw,dh])

model = misc_nets.UnetModel(nfilters=48, input_size=input_size + (1,))
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy',
              metrics=['accuracy'])
print(model.summary())

#model.fit_generator(myGene, steps_per_epoch=200,epochs=1)
#v=model.predict(testGene,verbose=1,steps=20)
#plot3Dtensor(v)

# Create the base model from the pre-trained model MobileNet V2
# Load MobileNetV2:
#base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                           #include_top=False, weights='imagenet')

#image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
# train_data_gen = image_gen.flow_from_directory(batch_size=5,
#                                                directory='C:/work/titan/TitanTestSets/FOV_Rings_1600',
#                                                shuffle=True,
#                                                target_size=(256, 256))
# augmented_images = [train_data_gen[0][0][0] for i in range(25)]

