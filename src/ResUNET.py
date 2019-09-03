import tensorflow as tf


def upsample(inputs, numFilters, kernelSize):
	#return tf.keras.backend.resize_volumes(inputs, 2, 2, 2, "channels_last")
	return tf.layers.conv3d_transpose(inputs, numFilters, kernelSize, 2, 'SAME', use_bias=False, kernel_initializer='he_normal')


def downsample(inputs):
	return tf.nn.avg_pool3d(inputs, [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], 'SAME')


def residualBlockDown(inputs, numFilters, kernelSize, isTraining):
	'''
		First part
	'''
	conv1 = tf.layers.conv3d(inputs, numFilters, kernelSize, 1, 'SAME', use_bias=False, kernel_initializer='he_normal')
	#tf.layers.batch_normalization(conv1, training=isTraining)
	relu1 = tf.nn.leaky_relu(conv1)
	print(relu1)

	conv2 = tf.layers.conv3d(relu1, numFilters, kernelSize, 1, 'SAME', use_bias=False, kernel_initializer='he_normal')
	print(conv2)

	resizedConvInput = downsample(conv2)
	print(resizedConvInput)

	'''
		Second part
	'''
	resizedInput = downsample(inputs)
	print(resizedInput)

	convInput = tf.layers.conv3d(resizedInput, numFilters, kernelSize, 1, 'SAME', use_bias=False, kernel_initializer='he_normal')
	#tf.layers.batch_normalization(conv2, training=isTraining)
	print(convInput)

	'''
		Skip connection
	'''
	output = resizedConvInput + convInput
	print(output)

	output = tf.nn.leaky_relu(output)
	print(str(output) + "\n")

	return output


def residualBlockUp(inputs, numFilters, kernelSize, isTraining):
	'''
		First part
	'''
	conv1 = tf.layers.conv3d(inputs, numFilters, kernelSize, 1, 'SAME', use_bias=False, kernel_initializer='he_normal')
	#tf.layers.batch_normalization(conv1, training=isTraining)
	relu1 = tf.nn.leaky_relu(conv1)
	print(relu1)

	#conv2 = tf.layers.conv3d(relu1, numFilters, kernelSize, 1, 'SAME', use_bias=False, kernel_initializer='he_normal')
	#print(conv2)

	resizedConvInput = upsample(conv1, numFilters, kernelSize)
	print(resizedConvInput)

	'''
		Second part
	'''
	resizedInput = upsample(inputs, numFilters, kernelSize)
	print(resizedInput)

	#convInput = tf.layers.conv3d(resizedInput, numFilters, kernelSize, 1, 'SAME', use_bias=False, kernel_initializer='he_normal')
	#tf.layers.batch_normalization(conv2, training=isTraining)
	#print(convInput)

	'''
		Skip connection
	'''
	output = resizedConvInput + resizedInput
	print(output)

	output = tf.nn.leaky_relu(output)
	print(str(output) + "\n")

	return output


def getGenerator(X, reuse=False, kernelSize=3):
	filters = [64, 128, 256, 512, 512, 512]

	with tf.variable_scope("generator", reuse=reuse):
		print("\n" + str(X))
		output = X

		# Encoder
		skips = []
		for numFilters in filters:
			output = residualBlockDown(output, numFilters, kernelSize, None)
			skips.append(output)

		# Decoder
		for numFilters, skip in zip(reversed(filters[:-1]), reversed(skips[:-1])):
			output = residualBlockUp(output, numFilters, kernelSize, None)
			output = tf.concat([output, skip], axis=4)

		output = residualBlockUp(output, numFilters, kernelSize, None)
		
		last = tf.layers.conv3d(output, 1, kernelSize, 1, 'SAME', use_bias=False, kernel_initializer='he_normal')
		print(last)
		print("\n")
		return last


def getDiscriminator(X, Y, reuse=False, kernelSize=4):
	filters = [64, 128, 256]

	with tf.variable_scope('discriminator', reuse=reuse):
		output = tf.concat([X, Y], axis=-1)
		print(output)

		for numFilters in filters:
			output = residualBlockDown(output, numFilters, 4, None)

		last = tf.layers.conv3d(output, 1, kernelSize, 1, 'SAME', use_bias=False, kernel_initializer='he_normal')
		print(last)

		return last