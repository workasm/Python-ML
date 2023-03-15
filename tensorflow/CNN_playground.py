
from __future__ import absolute_import, division, print_function, unicode_literals

from sympy import nfloat

from common import *
import dense_net
import misc_nets

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib
import numpy as np
#import cv_image_proc
# how to start computation
# runfile('CNN_playground.py', wdir='C:/work/tensorflow')
#serImage = "C:/work/titan/TitanTestSets/Sirius/2013-02-21__14-47-53_Wilhelm_Oliver_0005799936_OD.ser"

ANGULAR_RES = 360//2
RADIAL_RES = 256//2
BATCH_SIZE = 5
EPOCHS = 16

def image_resizer(imageRef, imageLos):
    image = tf.concat([imageRef, imageLos], axis=-1)
    return tf.image.resize(image, [RADIAL_RES,ANGULAR_RES])

def image_decoder_OK(fileRef, fileLos):
    imRef = decode_png(fileRef)
    imLos = decode_png(fileLos)
    image = tf.concat([imRef, imLos], axis=-1)
    #return (image, 1)
    return (tf.image.resize(image, [RADIAL_RES, ANGULAR_RES]), 1)

def image_decoder_NOK(fileRef, fileLos):
    imRef = decode_png(fileRef)
    imLos = decode_png(fileLos)
    image = tf.concat([imRef, imLos], axis=-1)
    #return (image, 0)
    return (tf.image.resize(image, [RADIAL_RES, ANGULAR_RES]), 0)

data_dir = pathlib.Path("C:/work/titan/TitanTestSets/SiriusUnrolled")
#image_list = list(data_dir.glob('*.ser'))
list_ref = tf.data.Dataset.list_files(str(data_dir/'*Ref.png'), shuffle=False)
list_los = tf.data.Dataset.list_files(str(data_dir/'*Los.png'), shuffle=False)
image_ref = list_ref.map(decode_png)
image_los = list_los.map(decode_png)
#images_OK = tf.data.Dataset.zip((image_ref, image_los))
images_OK = tf.data.Dataset.zip((list_ref, list_los))
data_size = len(list(list_ref))*2

list_los2 = list_los.shuffle(data_size//2)
#images_NOK = tf.data.Dataset.zip((image_ref, image_los2))
images_NOK = tf.data.Dataset.zip((list_ref, list_los2))

#images_OK = images_OK.map(image_resizer).map(lambda x: (x,1))
#images_NOK = images_NOK.map(image_resizer).map(lambda x: (x,0))
images_OK = images_OK.map(image_decoder_OK)
images_NOK = images_NOK.map(image_decoder_NOK)
image_set = images_OK.concatenate(images_NOK)
#image_set = image_set.map(decode_png)
image_set = image_set.cache('C:/work/tensorflow/train_cache').shuffle(data_size)

# for f,l in image_set.take(25):
#   print(f[0].numpy(),f[1].numpy())

train_size = int(0.7 * data_size)
#val_size = int(0.15 * data_size)
test_size = int(0.3 * data_size)
STEPS_PER_EPOCH = train_size // BATCH_SIZE

train_set = image_set.take(train_size)
test_set = image_set.skip(train_size).batch(BATCH_SIZE)
train_set = train_set.batch(BATCH_SIZE).repeat()
train_set = train_set.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

train_batch = next(iter(test_set))
#z1,z2 = tf.split(train_batch[0], 2, -1)
#plotImages(z1.numpy())

def OnTheFlyDataGen(aug_dict):
    image_datagen = ImageDataGenerator(**aug_dict)

    #here we can just call our module to generate artifact mask for an image!!!
    #and feed it directly to the model...

    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)


def mergeModels():
    model_input = tfkl.Input(shape=(RADIAL_RES,ANGULAR_RES,2))
    a, b = tf.split(model_input, 2, axis=-1)
    mc1 = misc_nets.MixConvResModel(nflts=16,input_size=(RADIAL_RES, ANGULAR_RES, 1), includeTop=False)
    mc2 = misc_nets.MixConvResModel(nflts=16,input_size=(RADIAL_RES, ANGULAR_RES, 1), includeTop=False)
    ma = mc1(a)
    mb = mc2(b)
    x = tfkl.Concatenate(axis=-1)([ma, mb])

    #flat1 = tfkl.Flatten()(z)
    #dense1 = tfkl.Dense(16, activation='relu')(flat1)
    #dense2 = tfkl.Dense(256, activation='relu')(dense1)
    #x = tfkl.Dense(1, activation='sigmoid')(x)
    x = tfkl.Activation('relu')(x)
    x = tfkl.GlobalAveragePooling2D()(x)
    x = tfkl.Dense(1, activation='sigmoid')(x)

    modelC = tf.keras.Model(inputs = model_input, outputs = x)
    return modelC

mc = dense_net.create(1, (RADIAL_RES,ANGULAR_RES,2), depth=19, nb_dense_block=3, growth_rate=16, nb_filter=16)
#mc = misc_nets.MixConvResModel(nflts=32, input_size=(RADIAL_RES, ANGULAR_RES,2))
#mc = mergeModels()
print(mc.summary())
#tf.keras.utils.plot_model(mc, show_shapes=True)
#tf.keras.optimizers.RMSprop(lr=base_learning_rate),
mc.compile(optimizer=tf.optimizers.Adam(),
           loss='binary_crossentropy', metrics=['accuracy'])

mc_history = mc.fit(train_set, epochs=EPOCHS,
                          steps_per_epoch=train_size//BATCH_SIZE,
                          # validation_steps=VALIDATION_STEPS,
                          validation_data=test_set)
                          # callbacks=[DisplayCallback()])

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

# manual training step with gradientTape:
def train_step(images, labels):
    with tf.GradientTape() as tape:
        logits = mnist_model(images, training=True)

        # Add asserts to check the shape of the output.
        tf.debugging.assert_equal(logits.shape, (32, 10))

        loss_value = loss_object(labels, logits)

    loss_history.append(loss_value.numpy().mean())
    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))


def train(epochs):
    for epoch in range(epochs):
        for (batch, (images, labels)) in enumerate(dataset):
            train_step(images, labels)
        print('Epoch {} finished'.format(epoch))

