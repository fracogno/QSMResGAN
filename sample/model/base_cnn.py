import tensorflow as tf


class BaseCNN(tf.keras.Model):

    def __init__(self):
        super(BaseCNN, self).__init__()

    def CNN_layer_3D(self, num_filters, k_size, stride, initializer, use_bias, upsampling=False, apply_batch_norm=False, dropout_rate=0.0, activation=None):
        result = tf.keras.Sequential()

        # Decide whether downsampling or upsampling
        if upsampling:
            result.add(tf.keras.layers.Conv3DTranspose(num_filters, k_size, strides=stride, padding='SAME', kernel_initializer=initializer, use_bias=use_bias))
        else:
            result.add(tf.keras.layers.Conv3D(num_filters, k_size, strides=stride, padding='SAME', kernel_initializer=initializer, use_bias=use_bias))

        result.add(tf.keras.layers.BatchNormalization()) if apply_batch_norm else None
        result.add(tf.keras.layers.Dropout(dropout_rate)) if float(dropout_rate) > 0. else None
        result.add(activation) if not activation is None else None

        return result
