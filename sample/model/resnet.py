import tensorflow as tf

from model import base_cnn


class ResBlock(base_cnn.BaseCNN):
    def __init__(self, num_filters, kernel_size, stride, initializer, use_bias, upsampling=False, batch_norm=False, dropout_rate=0.0, activation=tf.keras.layers.ReLU()):
        super(ResBlock, self).__init__()

        # https://www.tensorflow.org/tutorials/customization/custom_layers RESNET LAYER
        self.conv1a = self.CNN_layer_3D(num_filters, kernel_size, stride, initializer, use_bias, upsampling, batch_norm, dropout_rate, activation)
        self.conv2 = self.CNN_layer_3D(num_filters, kernel_size, 1, initializer, use_bias, False, batch_norm, dropout_rate)

        self.conv1b = self.CNN_layer_3D(num_filters, 1, stride, initializer, use_bias, upsampling, batch_norm, dropout_rate)

    def call(self, input_tensor, training=False):
        # First part
        x = self.conv1a(input_tensor, training=training)
        x = self.conv2(x, training=training)

        # Second part
        x2 = self.conv1b(input_tensor, training=training)

        # Residual connection
        x += x2

        return tf.nn.relu(x)
