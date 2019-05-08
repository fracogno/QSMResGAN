import tensorflow as tf
from tensorflow.python.ops import array_ops

def getNetwork(features, filter_scale=4, reuse=False):
    
    with tf.variable_scope("generator", reuse=reuse):

        conv1 = tf.keras.layers.Conv3D(int(64 / filter_scale), kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                       activation='relu',
                                       padding='same', name='conv1', kernel_initializer='he_normal')(features)

        conv1_1 = tf.keras.layers.Conv3D(int(64 / filter_scale), kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                         activation='relu',
                                         padding='same', name='conv2', kernel_initializer='he_normal')(conv1)

        pool1 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), name='pool1')(conv1_1)

        conv2 = tf.keras.layers.Conv3D(int(128 / filter_scale), kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                       activation='relu',
                                       padding='same', name='conv3', kernel_initializer='he_normal')(pool1)

        conv2_1 = tf.keras.layers.Conv3D(int(128 / filter_scale), kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                         activation='relu',
                                         padding='same', name='conv4', kernel_initializer='he_normal')(conv2)

        pool2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), name='pool2')(conv2_1)

        conv3 = tf.keras.layers.Conv3D(int(256 / filter_scale), kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                       activation='relu',
                                       padding='same', name='conv5', kernel_initializer='he_normal')(pool2)

        conv3_1 = tf.keras.layers.Conv3D(int(256 / filter_scale), kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                         activation='relu',
                                         padding='same', name='conv6', kernel_initializer='he_normal')(conv3)

        pool3 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), name='pool3')(conv3_1)

        conv4 = tf.keras.layers.Conv3D(int(512 / filter_scale), kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                       activation='relu',
                                       padding='same', name='conv7', kernel_initializer='he_normal')(pool3)

        conv4_1 = tf.keras.layers.Conv3D(int(512 / filter_scale), kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                         activation='relu',
                                         padding='same', name='conv8', kernel_initializer='he_normal')(conv4)

        pool4 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), name='pool4')(conv4_1)

        conv5 = tf.keras.layers.Conv3D(int(1024 / filter_scale), kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                       activation='relu',
                                       padding='same', name='conv9', kernel_initializer='he_normal')(pool4)

        conv5_1 = tf.keras.layers.Conv3D(int(1024 / filter_scale), kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                         activation='relu',
                                         padding='same', name='conv10', kernel_initializer='he_normal')(conv5)



        up6 = tf.concat(
            [tf.keras.layers.Conv3DTranspose(int(512 / filter_scale), (2, 2, 2), strides=(2, 2, 2), padding='same',
                                             activation='relu', name='up_conv1', kernel_initializer='he_normal')(
                conv5_1), conv4_1],
            axis=-1)
        conv6 = tf.keras.layers.Conv3D(int(512 / filter_scale), kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                       activation='relu',
                                       padding='same', name='conv11', kernel_initializer='he_normal')(up6)
        conv6 = tf.keras.layers.Conv3D(int(512 / filter_scale), kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                       activation='relu',
                                       padding='same', name='conv12', kernel_initializer='he_normal')(conv6)

        up7 = tf.concat(
            [tf.keras.layers.Conv3DTranspose(int(256 / filter_scale), (2, 2, 2), strides=(2, 2, 2), padding='same',
                                             activation='relu', name='up_conv2', kernel_initializer='he_normal')(
                conv6), conv3_1], axis=-1)
        conv7 = tf.keras.layers.Conv3D(int(256 / filter_scale), kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                       activation='relu',
                                       padding='same', name='conv13', kernel_initializer='he_normal')(up7)
        conv7 = tf.keras.layers.Conv3D(int(256 / filter_scale), kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                       activation='relu',
                                       padding='same', name='conv14', kernel_initializer='he_normal')(conv7)

        up8 = tf.concat(
            [tf.keras.layers.Conv3DTranspose(int(128 / filter_scale), (2, 2, 2), strides=(2, 2, 2), padding='same',
                                             activation='relu', name='up_conv3', kernel_initializer='he_normal')(
                conv7), conv2_1], axis=-1)
        conv8 = tf.keras.layers.Conv3D(int(128 / filter_scale), kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                       activation='relu',
                                       padding='same', name='conv15', kernel_initializer='he_normal')(up8)
        conv8 = tf.keras.layers.Conv3D(int(128 / filter_scale), kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                       activation='relu',
                                       padding='same', name='conv16', kernel_initializer='he_normal')(conv8)

        up9 = tf.concat(
            [tf.keras.layers.Conv3DTranspose(int(64 / filter_scale), (2, 2, 2), strides=(2, 2, 2), padding='same',
                                             activation='relu', name='up_conv4', kernel_initializer='he_normal')(
                conv8), conv1_1], axis=-1)
        conv9 = tf.keras.layers.Conv3D(int(64 / filter_scale), kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                       activation='relu',
                                       padding='same', name='conv17', kernel_initializer='he_normal')(up9)
        conv9 = tf.keras.layers.Conv3D(int(64 / filter_scale), kernel_size=(3, 3, 3), strides=(1, 1, 1),
                                       activation='relu',
                                       padding='same', name='conv18', kernel_initializer='he_normal')(conv9)

        adaptation_layer_1 = tf.keras.layers.Conv3D(kernel_size=1, filters=128, strides=(1, 1, 1), name='output_layer',
                                              kernel_initializer='he_normal', activation=None)(conv9)

        adaptation_layer_2 = tf.keras.layers.Conv3D(kernel_size=1, filters=64, strides=(1, 1, 1), name='output_layer',
                                              kernel_initializer='he_normal', activation=None)(adaptation_layer_1)

        output_layer = tf.keras.layers.Conv3D(kernel_size=1, filters=1, strides=(1, 1, 1), name='output_layer',
                                              kernel_initializer='he_normal', activation=None)(adaptation_layer_2 )

        output_layer = tf.add(output_layer, features)
    
    
    return output_layer

