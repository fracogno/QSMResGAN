import tensorflow as tf


def convLayer(x, filters, kernelSize, stride, padding, use_bias, upsample=False):

	if not upsample:
		x = tf.layers.conv3d(x, filters, kernelSize, stride, padding, use_bias=use_bias) #, kernel_initializer='he_normal')
	else:
		x = tf.layers.conv3d_transpose(x, filters, kernelSize, stride, padding, use_bias=use_bias) #, kernel_initializer='he_normal')

	#x = tf.layers.batch_normalization(out, training=isTraining)
	x = tf.nn.relu(x)
	print(x)

	return x


def basicBlock(x, filters, kernelSize, stride, use_bias):
	residual = x

	out = convLayer(x, filters, kernelSize, stride, 'SAME', use_bias=use_bias) #, kernel_initializer='he_normal')

	out = tf.layers.conv3d(out, filters, kernelSize, 1, 'SAME', use_bias=use_bias) #, kernel_initializer='he_normal')
	#out = tf.layers.batch_normalization(out, training=isTraining)
	print(out)

	if stride > 1:
		residual = tf.layers.conv3d(x, filters, 1, stride, 'SAME', use_bias=use_bias) #, kernel_initializer='he_normal')
		#residual = tf.layers.batch_normalization(residual, training=isTraining)

	# Skip connection
	out += residual
	print(out)

	# Final activation
	out = tf.nn.relu(out)
	print(out)

	return out


def Encoder(x, filters, kernelSize, stride, use_bias):
	x = basicBlock(x, filters, kernelSize, stride, use_bias)
	x = basicBlock(x, filters, kernelSize, 1, use_bias)
	print()

	return x


def Decoder(x, filters, kernelSize, stride, use_bias):
	x = convLayer(x, filters//2, 1, 1, 'SAME', use_bias=use_bias) 
	x = convLayer(x, filters//2, kernelSize, stride, 'SAME', use_bias=use_bias, upsample=True) # kernel_initializer='he_normal')
	x = convLayer(x, filters, 1, 1, 'SAME', use_bias=use_bias)
	print()

	return x


def getGenerator(x, reuse=False, kernelSize=3, use_bias=False):
	print(x)
	############## TO DO : leaky relu, concat instead of ADD, remove one layer and put stride 2, kernel initial, BN

	with tf.variable_scope("generator", reuse=reuse):
		# Initial layer
		x = convLayer(x, 64, 7, 2, 'SAME', use_bias=use_bias)
		x = tf.layers.max_pooling3d(x, 3, 2, "SAME")
		print(str(x) + "\n")

		# Encoding
		e1 = Encoder(x, 64, kernelSize, 1, use_bias)
		e2 = Encoder(e1, 128, kernelSize, 2, use_bias)
		e3 = Encoder(e2, 256, kernelSize, 2, use_bias)
		e4 = Encoder(e3, 512, kernelSize, 2, use_bias)
		e5 = Encoder(e4, 512, kernelSize, 2, use_bias)

		# Decoding
		d5 = e4 + Decoder(e5, 512, kernelSize, 2, use_bias)
		d4 = e3 + Decoder(d5, 256, kernelSize, 2, use_bias)
		d3 = e2 + Decoder(d4, 128, kernelSize, 2, use_bias)
		d2 = e1 + Decoder(d3, 64, kernelSize, 2, use_bias)
		d1 = x + Decoder(d2, 64, kernelSize, 1, use_bias)

		# Final layers
		y = convLayer(d1, 32, kernelSize, 2, 'SAME', use_bias=use_bias, upsample=True)
		y = convLayer(y, 32, kernelSize, 1, 'SAME', use_bias=use_bias)
		last = tf.layers.conv3d_transpose(y, 1, 2, 2, 'SAME', use_bias=use_bias) #, kernel_initializer='he_normal')
		print(last)

		return last


def getDiscriminator(X, Y, reuse=False, kernelSize=4):
	filters = [64, 128, 256]

	with tf.variable_scope('discriminator', reuse=reuse):
		output = tf.concat([X, Y], axis=-1)
		print(output)

		for numFilters in filters:
			output = residualBlockDown(output, numFilters, 4, None)

		last = tf.layers.conv3d(output, 1, kernelSize, 1, 'SAME')#, use_bias=False, kernel_initializer='he_normal')
		print(last)

		return last