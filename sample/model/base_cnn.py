import tensorflow as tf


class BaseCNN(tf.keras.Model):

    def __init__(self):
        super(BaseCNN, self).__init__()

    def CNN_layer_3D(self, num_filters, kernel_size, stride, initializer, use_bias, upsampling, apply_batch_norm, dropout_rate, activation):
        result = tf.keras.Sequential()

        # Decide whether downsampling or upsampling
        if upsampling:
            result.add(tf.keras.layers.Conv3DTranspose(num_filters, kernel_size, strides=stride, padding='SAME', kernel_initializer=initializer, use_bias=use_bias))
        else:
            result.add(tf.keras.layers.Conv3D(num_filters, kernel_size, strides=stride, padding='SAME', kernel_initializer=initializer, use_bias=use_bias))

        result.add(tf.keras.layers.BatchNormalization()) if apply_batch_norm else None
        result.add(tf.keras.layers.Dropout(dropout_rate)) if dropout_rate > 0.0 else None
        result.add(tf.keras.layers.ReLU()) if activation else None

        return result
