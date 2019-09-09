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


def resnetBlock(inputs, filters, kernelSize, stride, padding, use_bias, use_dropout=False):

	x = convLayer(inputs, filters, kernelSize, stride, padding, use_bias)

	if use_dropout:
		x = tf.layers.dropout(x, rate=0.5, training=True)

	x = convLayer(x, filters, kernelSize, stride, padding, use_bias, relu=False)


	skip = inputs + x
	print(str(skip) + "\n")

	return skip
 

def getGenerator(x, reuse=False, kernelSize=3, use_bias=False):
	print(x)
	'''
		TO DO : 
			1 - leaky relu
			2 - Add long skip conn
			3 - Make it deeper
			4 - kernel initial
			5 - BN
			6 - bias=True
	'''
	with tf.variable_scope("generator", reuse=reuse):
		ngf = 64
		nBlockGen = 9

		x = convLayer(x, ngf, 7, 1, "SAME", use_bias)

		n_downsampling = 2
		for i in range(n_downsampling):
			mult = 2**i
			x = convLayer(x, ngf*mult*2, kernelSize, 2, "SAME", use_bias)

		# Resnet blocks
		mult = 2**n_downsampling
		for i in range(nBlockGen):
			x = resnetBlock(x, ngf*mult, kernelSize, 1, "SAME", use_bias)

		for i in range(n_downsampling):
			mult = 2**(n_downsampling - i)
			x = convLayer(x, int(ngf*mult / 2), kernelSize, 2, "SAME", use_bias, upsample=True)

		x = tf.layers.conv3d(x, 1, 7, 1, "SAME", use_bias=use_bias) #, kernel_initializer='he_normal')
		print(x)

		return x