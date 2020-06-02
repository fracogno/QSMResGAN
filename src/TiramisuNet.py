import tensorflow as tf



def BN_ReLU_Conv(x, n_filters, filter_size):

	#x = tf.layers.batch_normalization(training=training)
	x = tf.nn.relu(x)
	print(x)

	x = tf.layers.conv3d(x, n_filters, filter_size, padding='SAME', kernel_initializer='he_uniform')
	print(x)

	# x = tf.layers.dropout(x, rate=0.2, training=training)

	return x


def TransitionDown(x, n_filters):
	l = BN_ReLU_Conv(x, n_filters, 1)
	l = tf.layers.max_pooling3d(l, 2, 2)#, padding='SAME')
	print(str(l) + "\n")

	return l


def TransitionUp(skip_connection, block_to_upsample, n_filters, filter_size):

	# Upsample
	l = tf.concat(block_to_upsample, axis=-1)
	print(l)
	l = tf.layers.conv3d_transpose(l, n_filters, filter_size, 2, padding='SAME', kernel_initializer='he_uniform')
	print(l)

	# Concat skip
	l = tf.concat([l, skip_connection], axis=-1)
	print(str(l) + "\n")

	return l



def getGenerator(x, reuse=False):

	growth_rate = 16
	n_pool = 5
	n_layers_per_block = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
	n_filters = 48
	assert(len(n_layers_per_block) == 2 * n_pool + 1)

	with tf.variable_scope("generator", reuse=reuse):
		print(x)

		#####################
		# First Convolution #
		#####################
		stack = tf.layers.conv3d(x, n_filters, 3, padding='SAME', kernel_initializer='he_uniform')
		print(stack)

		#####################
		# Downsampling path #
		#####################
		skip_connection_list = []
		for i in range(n_pool):

			# Dense block
			for j in range(n_layers_per_block[i]):
				l = BN_ReLU_Conv(stack, growth_rate, 3)
				stack = tf.concat([stack, l], axis=-1)
				n_filters += growth_rate

			skip_connection_list.append(stack)
			stack = TransitionDown(stack, n_filters)

		skip_connection_list = skip_connection_list[::-1]


		#####################
		#     Bottleneck    #
		#####################
		block_to_upsample = []
		for j in range(n_layers_per_block[n_pool]):
			l = BN_ReLU_Conv(stack, growth_rate, 3)
			block_to_upsample.append(l)
			stack = tf.concat([stack, l], axis=-1)
			print(stack)

		#####################
		# Upsampling path #
		#####################
		for i in range(n_pool):
			n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
			stack = TransitionUp(skip_connection_list[i], block_to_upsample, n_filters_keep, 3)

			block_to_upsample = []
			for j in range(n_layers_per_block[ n_pool + i + 1 ]):
				l = BN_ReLU_Conv(stack, growth_rate, 3)
				block_to_upsample.append(l)
				stack = tf.concat([stack, l], axis=-1)
			block_to_upsample = tf.concat(block_to_upsample, axis=-1)


		x = tf.layers.conv3d(block_to_upsample, 1, 1, 1, "SAME", kernel_initializer='he_normal')
		print(str(x) + "\n\n")

		return x