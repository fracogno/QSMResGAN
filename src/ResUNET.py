import tensorflow as tf


def upsample(inputs):
	return tf.keras.backend.resize_volumes(inputs, 2, 2, 2, "channels_last")


def downsample(inputs):
	return tf.nn.avg_pool3d(inputs, [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], 'SAME')


'''def residualBlock(inputs, numFilters, kernelSize, isTraining, encoding=True):
	print(inputs)

	# First part
	conv1 = tf.layers.conv3d(inputs, numFilters, kernelSize, 1, 'SAME', use_bias=False, kernel_initializer='he_normal')
	#bn1 = tf.layers.batch_normalization(conv1, training=isTraining)
	relu1 = tf.nn.leaky_relu(conv1)
	print(relu1)

	# Second part
	conv2 = tf.layers.conv3d(relu1, numFilters, kernelSize, 1, 'SAME', use_bias=False, kernel_initializer='he_normal')
	#bn2 = tf.layers.batch_normalization(conv2, training=isTraining)
	print(conv2)

	# Concat
	#skipConnection = tf.concat([inputs, conv2], axis=4)
	skipConnection = inputs + conv2
	reluSkip = tf.nn.leaky_relu(skipConnection)
	print(reluSkip)

	if encoding:
		output = downsample(reluSkip)
	else:
		output = upsample(reluSkip)

	print(output)
	print()

	return output'''



def residualBlock(inputs, numFilters, kernelSize, isTraining, encoding=True):
	print(inputs)
	
	'''
		First part
	'''
	# Convolve input
	conv1 = tf.layers.conv3d(inputs, numFilters, kernelSize, 1, 'SAME', use_bias=False, kernel_initializer='he_normal')
	#bn1 = tf.layers.batch_normalization(conv1, training=isTraining)
	relu1 = tf.nn.leaky_relu(conv1)
	print(relu1)

	# Convolve result of conv input
	conv2 = tf.layers.conv3d(relu1, numFilters, kernelSize, 1, 'SAME', use_bias=False, kernel_initializer='he_normal')
	print(conv2)

	# Resize convolved input
	resizedConvInput = downsample(conv2) if encoding else upsample(conv2)
	print(resizedConvInput)

	'''
		Second part
	'''
	# Resize original input
	resizedInput = downsample(inputs) if encoding else upsample(inputs)
	print(resizedInput)

	# Convolve resized input
	convInput = tf.layers.conv3d(resizedInput, numFilters, kernelSize, 1, 'SAME', use_bias=False, kernel_initializer='he_normal')
	#bn2 = tf.layers.batch_normalization(conv2, training=isTraining)
	print(convInput)

	# Skip connection
	output = resizedConvInput + convInput
	print(output)

	output = tf.nn.leaky_relu(output)
	print(output)
	print()

	return output


def getGenerator(X, reuse=False):
	filters = [64, 128, 256, 512, 512, 512]

	with tf.variable_scope("generator", reuse=reuse):
		print("\n" + str(X))
		output = X

		# Encoder
		skips = []
		skips.append(output)
		for numFilters in filters:
			output = residualBlock(output, numFilters, 4, None)
			skips.append(output)

		# Decoder
		skips = reversed(skips[:-1])
		for numFilters, skip in zip(reversed(filters), skips):
			output = residualBlock(output, numFilters, 4, None, False)
			output = tf.concat([output, skip], axis=4)

		last = tf.layers.conv3d(output, 1, 4, 1, 'SAME', use_bias=False, kernel_initializer='he_normal')
		print(last)

		return last


def getDiscriminator(X, Y, reuse=False):
	filters = [64, 128]

	with tf.variable_scope('discriminator', reuse=reuse):
		output = tf.concat([X, Y], axis=-1)
		print(output)

		for numFilters in filters:
			output = residualBlock(output, numFilters, 4, None)

		conv = tf.layers.conv3d(output, 256, 4, strides=1, use_bias=False, kernel_initializer='he_normal')
		#bn = tf.contrib.layers.batch_norm(conv, decay=0.9, is_training=True, updates_collections=None, epsilon=1e-5, scale=True)
		lrelu = tf.nn.leaky_relu(conv)
		print(lrelu)

		last = tf.layers.conv3d(lrelu, 1, 4, 1, 'SAME', use_bias=False, kernel_initializer='he_normal')
		print(last)

		return last