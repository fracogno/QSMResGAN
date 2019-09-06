import tensorflow as tf


def residualBlockDown(inputs, numFilters, kernelSize, isTraining):
	''' First part '''
	conv1 = tf.layers.conv3d(inputs, numFilters, kernelSize, 2, 'SAME')#, use_bias=False, kernel_initializer='he_normal')
	#tf.layers.batch_normalization(conv1, training=isTraining)
	relu1 = tf.nn.leaky_relu(conv1)
	print(relu1)

	conv2 = tf.layers.conv3d(relu1, numFilters, kernelSize, 1, 'SAME')#, use_bias=False, kernel_initializer='he_normal')
	#tf.layers.batch_normalization(relu1, training=isTraining)
	print(conv2)

	''' Second part '''
	conv3 = tf.layers.conv3d(inputs, numFilters, kernelSize, 2, 'SAME')#, use_bias=False, kernel_initializer='he_normal')
	#tf.layers.batch_normalization(convInput, training=isTraining)
	print(conv3)

	'''	Skip connection	'''
	output = conv2 + conv3
	print(output)

	output = tf.nn.leaky_relu(output)
	print(str(output) + "\n")

	return output


def residualBlockUp(inputs, numFilters, kernelSize, isTraining):
	''' First part '''
	conv1 = tf.layers.conv3d_transpose(inputs, numFilters, kernelSize, 2, 'SAME')#, use_bias=False, kernel_initializer='he_normal')
	#tf.layers.batch_normalization(conv1, training=isTraining)
	relu1 = tf.nn.leaky_relu(conv1)
	print(relu1)

	conv2 = tf.layers.conv3d(relu1, numFilters, kernelSize, 1, 'SAME')#, use_bias=False, kernel_initializer='he_normal')
	print(conv2)
	
	''' Second part '''
	conv3 = tf.layers.conv3d_transpose(inputs, numFilters, kernelSize, 2, 'SAME')#, use_bias=False, kernel_initializer='he_normal')
	#tf.layers.batch_normalization(conv3, training=isTraining)
	print(conv3)

	# Skip connection
	output = conv2 + conv3
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
			print(output)

		output = residualBlockUp(output, numFilters, kernelSize, None)
		
		last = tf.layers.conv3d(output, 1, kernelSize, 1, 'SAME')#, use_bias=False, kernel_initializer='he_normal')
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

		last = tf.layers.conv3d(output, 1, kernelSize, 1, 'SAME')#, use_bias=False, kernel_initializer='he_normal')
		print(last)

		return last
