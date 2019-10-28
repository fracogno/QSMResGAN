import tensorflow as tf
import numpy as np
import src.ResUNet as network, src.utilities as utils
import pickle
import nibabel as nib

# Paths
basePath = "/scratch/cai/deepQSMGAN/"
#basePath ="/home/francesco/UQ/deepQSMGAN/"
checkpointPath = "ckp_20191017_2348_shapes_shape64_ex100_2019_08_30/"
checkpointPath = "ckp_20191022_1335_shapes_shape64_ex100_2019_08_30/"

# Get data
X, Y, masks = utils.loadChallengeData(basePath + "data/QSM_Challenge2_download_stage2/DatasetsStep2/")
X_padded, originalShape, valuesSplit = utils.addPadding(X, 256)
X_tensor = tf.placeholder(tf.float32, shape=[None, X_padded.shape[1], X_padded.shape[2], X_padded.shape[3], X_padded.shape[4]])
is_train = tf.placeholder(tf.bool, name='is_train')

# Network
Y_generated = network.getGenerator(X_tensor)    

num_metrics = len(utils.getMetrics(Y, X, masks))
with tf.Session() as sess:

	for i in range(num_metrics):
		saver = tf.train.Saver()
		sess.run(tf.global_variables_initializer())
		saver.restore(sess, basePath + checkpointPath + "model-metric" + str(i))
		
		# Predict from network
		predicted = []
		for i in range(len(X_padded)):
			singlePrediction = sess.run(Y_generated, feed_dict={ X_tensor : [X_padded[i]], is_train : False })
			predicted.append(singlePrediction[0])

		predicted = utils.removePadding(np.array(predicted), originalShape, valuesSplit)
		predicted = utils.applyMaskToVolume(predicted, masks)

		# Calculate metrics over validation and save it
		metrics = utils.getMetrics(Y, predicted, masks)
		print(metrics)

		for j in range(len(predicted)):
			utils.saveNii(predicted[j], basePath + "data/deepQSMResGAN/out-metric" + str(i) + "-vol-" + str(j))
