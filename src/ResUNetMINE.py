import tensorflow as tf


def convLayer(x, filters, kernelSize, stride, padding, use_bias, relu=True, upsample=False):

	if not upsample:
		x = tf.layers.conv3d(x, filters, kernelSize, stride, padding, use_bias=use_bias) #, kernel_initializer='he_normal')
	else:
		x = tf.layers.conv3d_transpose(x, filters, kernelSize, stride, padding, use_bias=use_bias) #, kernel_initializer='he_normal')

	#x = tf.layers.batch_normalization(out, training=isTraining)
	if relu:
		x = tf.nn.relu(x)
	print(x)

	return x


def block(x, filters, kernelSize, stride, use_bias, upsample=False):
	print(x)
	# First
	out = convLayer(x, filters, kernelSize, stride, 'SAME', use_bias=use_bias, upsample=upsample)
	out = convLayer(out, filters, kernelSize, 1, 'SAME', use_bias=use_bias, relu=False)

	# Second
	residual = convLayer(x, filters, 1, stride, 'SAME', use_bias=use_bias, relu=False, upsample=upsample)

	# Skip connection
	out += residual
	print(out)

	out = tf.nn.relu(out)
	print(str(out) + "\n")

	return out	


def getGenerator(x, reuse=False, kernelSize=3, use_bias=False):
	print(x)

	with tf.variable_scope("generator", reuse=reuse):

		x = convLayer(x, 32, kernelSize, 1, "SAME", use_bias) # 64x64x32
		print()

		e1 = block(x, 64, kernelSize, 2, use_bias)	# 32x32x64
		e2 = block(e1, 128, kernelSize, 2, use_bias) # 16x16x128
		e3 = block(e2, 256, kernelSize, 2, use_bias) # 8x8x256

		b1 = block(e3, 512, kernelSize, 1, use_bias) # 8x8x512
		b2 = block(b1, 1024, kernelSize, 1, use_bias) # 8x8x1024
		b3 = block(b2, 512, kernelSize, 1, use_bias) # 8x8x512

		d3 = tf.concat([e2, block(b3, 256, kernelSize, 2, use_bias, upsample=True)], 4) # 16x16x128 (+) 16x16x256
		d2 = tf.concat([e1, block(d3, 128, kernelSize, 2, use_bias, upsample=True)], 4) # 32x32x64  (+) 32x32x128
		#d1 = tf.concat([e1, block(d2, 64, kernelSize, 2, use_bias, upsample=True)], 4)
		d1 = block(d2, 64, kernelSize, 2, use_bias, upsample=True) # 64x64x64

		x = tf.layers.conv3d(d1, 1, kernelSize, 1, "SAME", use_bias=use_bias) #, kernel_initializer='he_normal')
		print(str(x) + "\n")

		return x