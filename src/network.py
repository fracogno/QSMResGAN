import tensorflow as tf
import tensorflow as tf


def downsample(inputs, filters, size, apply_batchnorm=True):
    
    result = tf.layers.conv3d(inputs, filters, size, strides=2, padding='SAME', use_bias=False, 
                              kernel_initializer=tf.random_normal_initializer(0., 0.02))
    
    
    #if apply_batchnorm:
    #result = tf.contrib.layers.batch_norm(result, decay=0.9, is_training=True, updates_collections=None, epsilon=1e-5, scale=True)
    #result = tf.layers.batch_normalization(result, axis=4, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

    return tf.nn.leaky_relu(result)


def upsample(inputs, filters, size, apply_dropout=False):
    
    result = tf.layers.conv3d_transpose(inputs, filters, size, strides=2, padding='SAME', use_bias=False,
                                        kernel_initializer=tf.random_normal_initializer(0., 0.02))
    
    #result = tf.contrib.layers.batch_norm(result, decay=0.9, is_training=True, updates_collections=None, epsilon=1e-5, scale=True)

    if apply_dropout:
        result = tf.nn.dropout(result, keep_prob=0.5)

    return tf.nn.relu(result)
    
    
    
def getGenerator(X, reuse=False):
    filters = [64, 128, 256, 512, 512, 512]
    
    with tf.variable_scope("generator", reuse=reuse):
        output = X
        print(output)
        skips = []
        # Encoder
        for num_f in filters:
            output = downsample(output, num_f, 4, num_f != filters[0])
            skips.append(output)
            print(output)
        
        # Decoder
        skips = reversed(skips[:-1])
        for num_f, skip in zip(reversed(filters[:-1]), skips):
            output = upsample(output, num_f, 4, apply_dropout=num_f == 512)
            output = tf.concat([output, skip], axis=4)
            print(output)

        #output = upsample(output, 32, 4)
        #output = tf.concat([output, X], axis=4)
        #print(output)

        last = tf.layers.conv3d_transpose(output, 1, 4, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), activation=None)
        print(last)

        return last


def getDiscriminator(X, Y):

    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        initializer = tf.random_normal_initializer(0., 0.02)

        inputs = tf.concat([X, Y], axis=-1)
        down1 = downsample(inputs, 64, 4, False)
        down2 = downsample(down1, 128, 4)
        #down3 = downsample(down2, 256, 4)
        print(X)
        print(inputs)
        print(down1)
        print(down2)
        #print(down3)

        zero_pad1 = tf.keras.layers.ZeroPadding3D()(down2) 
        conv = tf.layers.conv3d(zero_pad1, 256, 4, strides=1, kernel_initializer=initializer, use_bias=False)
        #bn = tf.contrib.layers.batch_norm(conv, decay=0.9, is_training=True, updates_collections=None, epsilon=1e-5, scale=True)
        lrelu = tf.nn.leaky_relu(conv)
        print(lrelu)

        zero_pad2 = tf.keras.layers.ZeroPadding3D()(lrelu) 
        last = tf.layers.conv3d(zero_pad2, 1, 4, strides=1, kernel_initializer=initializer)
        print(last)

        return last

