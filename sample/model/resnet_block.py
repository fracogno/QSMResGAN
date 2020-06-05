import tensorflow as tf


class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, num_filters, k_size, use_bias, kernel_initializer):
        super(ResnetIdentityBlock, self).__init__(name='')

        self.conv2a = tf.keras.layers.Conv3D(num_filters, 1, strides=1, padding='same', use_bias=use_bias, kernel_initializer=kernel_initializer)
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv3D(num_filters, k_size, strides=1, padding='same', use_bias=use_bias, kernel_initializer=kernel_initializer)
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv3D(num_filters, 1, strides=1, padding='same', use_bias=use_bias, kernel_initializer=kernel_initializer)
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.leaky_relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.leaky_relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.leaky_relu(x)
