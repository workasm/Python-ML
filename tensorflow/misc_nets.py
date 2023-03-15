
from common import *

def identity_block(x, f, filters, s=1, conv_block=False):
    # Retrieve Filters
    F1, F2, F3 = filters
    x_shortcut = x
    init = tf.initializers.glorot_uniform(seed=0)
    # First component of main path
    x = tfkl.Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid',
                    kernel_initializer=init)(x)
    x = tfkl.BatchNormalization(axis=3)(x)
    x = tfkl.Activation('relu')(x)
    # Second component of main path
    x = tfkl.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
                    kernel_initializer=init)(x)
    x = tfkl.BatchNormalization(axis=3)(x)
    x = tfkl.Activation('relu')(x)
    # Third component of main path
    x = tfkl.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                    kernel_initializer=init)(x)
    x = tfkl.BatchNormalization(axis=3)(x)
    if conv_block:
        x_shortcut = tfkl.Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid',
                    kernel_initializer=init)(x_shortcut)
        x_shortcut = tfkl.BatchNormalization(axis=3)(x_shortcut)
    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    x = tfkl.Add()([x, x_shortcut])
    x = tfkl.Activation('relu')(x)
    return x

def MixConvResModel(nflts = 16, input_size = (256, 256, 2), includeTop = True) :

    init = tf.keras.initializers.he_normal()
    inputs = tfkl.Input(input_size)
    x = tfkl.Conv2D(nflts, 5, activation='relu', padding='same', kernel_initializer=init)(inputs)
    x = identity_block(x, 5, [nflts,nflts,nflts])

    x = tfkl.Conv2D(nflts*2, 5, activation='relu', padding='same', kernel_initializer=init)(x)
    x = tfkl.MaxPooling2D(pool_size=(2, 2))(x)

    x = identity_block(x, 5, [nflts*2, nflts*2, nflts*2])
    x = tfkl.Conv2D(nflts * 2, 3, activation='relu', padding='same', kernel_initializer=init)(x)

    x = identity_block(x, 3, [nflts * 2, nflts * 2, nflts * 2])
    x = tfkl.MaxPooling2D(pool_size=(2, 2))(x)

    x = tfkl.Conv2D(nflts * 4, 3, activation='relu', padding='same', kernel_initializer=init)(x)
    x = identity_block(x, 3, [nflts * 4, nflts * 4, nflts * 4])
    x = tfkl.MaxPooling2D(pool_size=(2, 2))(x)

    if includeTop:
        x = tfkl.Flatten()(x)
        x = tfkl.Dense(16, activation='relu')(x)
        x = tfkl.Dense(256, activation='relu')(x)
        x = tfkl.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

# input size: (batch, height, width, channels)
def ClassifyModel(nfilters = 16, input_size = (256,256,2), includeTop = True) :
    inputs = tfkl.Input(input_size)
    conv1 = tfkl.Conv2D(nfilters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    pool1 = tfkl.MaxPooling2D(pool_size=(2, 2))(conv1)
    #Dropout(0.2),
    conv2 = tfkl.Conv2D(nfilters*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    pool2 = tfkl.MaxPooling2D(pool_size=(2, 2))(conv2)
    #Dropout(0.2),
    conv3 = tfkl.Conv2D(nfilters*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    x = tfkl.MaxPooling2D(pool_size=(2, 2))(conv3)
    # Dropout(0.2),
    if includeTop:
        x = tfkl.Flatten()(x)
        x = tfkl.Dense(64, activation='relu')(x)
        x = tfkl.Dense(1, activation='sigmoid')(x)

    modelC = tf.keras.Model(inputs = inputs, outputs = x)
    return modelC

def UnetModel(nfilters = 64, input_size = (256,256,1)):

    inputs = tfkl.Input(input_size)
    conv1 = tfkl.Conv2D(nfilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = tfkl.Conv2D(nfilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = tfkl.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tfkl.Conv2D(nfilters*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = tfkl.Conv2D(nfilters*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = tfkl.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tfkl.Conv2D(nfilters*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = tfkl.Conv2D(nfilters*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = tfkl.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = tfkl.Conv2D(nfilters*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = tfkl.Conv2D(nfilters*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = tfkl.Dropout(0.5)(conv4)
    pool4 = tfkl.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = tfkl.Conv2D(nfilters*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = tfkl.Conv2D(nfilters*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = tfkl.Dropout(0.5)(conv5)

    up6 = tfkl.UpSampling2D(size=(2, 2))(drop5)
    up6 = tfkl.Conv2D(nfilters*8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up6)
    merge6 = tfkl.concatenate([drop4,up6], axis = 3)
    conv6 = tfkl.Conv2D(nfilters*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = tfkl.Conv2D(nfilters*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = tfkl.UpSampling2D(size=(2, 2))(conv6)
    up7 = tfkl.Conv2D(nfilters*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up7)
    merge7 = tfkl.concatenate([conv3,up7], axis = 3)
    conv7 = tfkl.Conv2D(nfilters*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = tfkl.Conv2D(nfilters*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = tfkl.UpSampling2D(size=(2, 2))(conv7)
    up8 = tfkl.Conv2D(nfilters*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up8)
    merge8 = tfkl.concatenate([conv2,up8], axis = 3)
    conv8 = tfkl.Conv2D(nfilters*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = tfkl.Conv2D(nfilters*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = tfkl.UpSampling2D(size=(2, 2))(conv8)
    up9 = tfkl.Conv2D(nfilters, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up9)
    merge9 = tfkl.concatenate([conv1,up9], axis = 3)
    conv9 = tfkl.Conv2D(nfilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = tfkl.Conv2D(nfilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = tfkl.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = tfkl.Conv2D(1, 1, activation = 'sigmoid', name = 'lastConv')(conv9)

    modelU = tf.keras.Model(inputs = inputs, outputs = conv10)
    return modelU

class ProbLayer(tf.keras.layers.Layer):

    def __init__(self, hidden_units, k_mixt):
        super(ProbLayer, self).__init__()
        self.hidden = tf.keras.layers.Dense(hidden_units,
               input_shape=(None, 1), activation=tf.nn.tanh
                , kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.alpha = tf.keras.layers.Dense(k_mixt, activation=lambda t: tf.nn.elu(t)+1)
        self.mu = tf.keras.layers.Dense(1, activation=None)
        self.sigma = tf.keras.layers.Dense(k_mixt, activation=tf.nn.softplus)
        self.dist = tfpl.DistributionLambda(
            make_distribution_fn=lambda t:
                tfd.MixtureSameFamily(
                    mixture_distribution=tfd.Categorical(probs=t[0]),
                    components_distribution=tfd.LogNormal(
                        loc=t[1], scale=t[2]))
        )

    def call(self, inputs):
        zx = self.hidden(inputs)
        p_alpha = self.alpha(zx)
        p_mu = self.mu(zx)
        p_sigma = self.sigma(zx)
        return self.dist((p_alpha, p_mu, p_sigma))

    def lossFunc(p_alpha, p_mu, p_sigma, y):
        gm = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=p_alpha),
            components_distribution=tfd.LogNormal(
                loc=p_mu,
                scale=p_sigma)
        )
        return -tf.reduce_sum(gm.log_prob(y))