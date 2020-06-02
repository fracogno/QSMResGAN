import src.utilities as utils
import tensorflow as tf

X_tensor, Y_tensor = utils.getTrainingDataTF("/home/francesco/Documents/University/UQ/QSMResGAN/data/shapes_shape64_ex100_2019_08_20", 1032, 1)

with tf.Session() as sess:
	X, Y = sess.run([X_tensor, Y_tensor])
	print(X.shape)
	print(Y.shape)
	for i in range(10):
		utils.saveNii(X[i], "/home/francesco/Documents/University/UQ/QSMResGAN/data/shapes_shape64_ex100_2019_08_20/X-" + str(i) + ".nii")
		utils.saveNii(Y[i], "/home/francesco/Documents/University/UQ/QSMResGAN/data/shapes_shape64_ex100_2019_08_20/Y-" + str(i) + ".nii")