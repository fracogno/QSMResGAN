import os
import tensorflow as tf
import numpy as np
import nibabel as nib
from datetime import datetime
import ResUNet as network, utilities as utils, loss

# Set GPU ??
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Paths
basePath = "/scratch/cai/QSMResGAN/"
dataPath = "dataset/shapes_shape64_ex100_2019_08_10"
checkpointsPath = basePath + "ckp_" + datetime.now().strftime("%Y%m%d_%H%M") + "_" + dataPath.split("/")[-1]

# Parameters
epochs = 2000
batchSize = 32
lr = 1e-4
beta1 = 0.5
L1_Weight = 100.0
labelSmoothing = 0.9

# Get data
X_tensor, Y_tensor = utils.getTrainingDataTF(basePath + dataPath, batchSize, epochs)
X_val, Y_val, mask_val = utils.loadChallengeData(basePath + "dataset/QSM_Challenge2_download_stage2/")
X_val_padded, originalShape, valuesSplit = utils.addPadding(X_val, (256,256,256))
X_val_tensor = tf.placeholder(tf.float32, shape=[None, X_val_padded.shape[1], X_val_padded.shape[2], X_val_padded.shape[3], X_val_padded.shape[4]], name='X_val')
is_train = tf.placeholder(tf.bool, name='is_train')

# Networks
Y_generated = network.getGenerator(X_tensor)    
Y_val_generated = network.getGenerator(X_val_tensor, True)

D_logits_real = network.getDiscriminator(X_tensor, Y_tensor)
D_logits_fake = network.getDiscriminator(X_tensor, Y_generated, True)

# Losses and optimizer
D_loss = loss.discriminatorLoss(D_logits_real, D_logits_fake, labelSmoothing)
G_loss, G_gan, G_L1 = loss.generatorLoss(D_logits_fake, Y_generated, Y_tensor, L1_Weight)
optimizer = loss.getOptimizer(lr, beta1, D_loss, G_loss)

# Tensorboard
halfVolume = int(X_tensor.shape[2]) // 2
train_summaries = [tf.summary.scalar('D_loss', D_loss), tf.summary.scalar('G_loss', G_gan), tf.summary.scalar('L1_loss', G_L1), \
                    tf.summary.image('in', X_tensor[:, :, :, halfVolume], max_outputs=1), tf.summary.image('out', Y_generated[:, :, :, halfVolume], max_outputs=1), \
                    tf.summary.image('label', Y_tensor[:, :, :, halfVolume], max_outputs=1)]
train_merged_summaries = tf.summary.merge(train_summaries)

rmseTensor = tf.placeholder(tf.float32, shape=())
ddrmseTensor = tf.placeholder(tf.float32, shape=())
val_summaries = [tf.summary.scalar('rmse', rmseTensor), tf.summary.scalar('ddrmse', ddrmseTensor)]
val_merged_summaries = tf.summary.merge(val_summaries)

# Array containing best metrics values
bestValMetrics = [1e6] * len(utils.getMetrics(Y_val, X_val, mask_val))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    # op to write logs to Tensorboard
    train_summary_writer = tf.summary.FileWriter(checkpointsPath + '/train', graph=tf.get_default_graph())
    val_summary_writer = tf.summary.FileWriter(checkpointsPath + '/val')

    globalStep = 0
    while True:
        try:
            _, summary = sess.run([optimizer, train_merged_summaries], feed_dict={ is_train : True })
            train_summary_writer.add_summary(summary, globalStep)

            # Check validation accuracy
            if globalStep % 250 == 0:
                predicted = []
                for i in range(len(X_val_padded)):
                    singlePrediction = sess.run(Y_val_generated, feed_dict={ X_val_tensor : [X_val_padded[i]], is_train : False })
                    predicted.append(singlePrediction[0])

                predicted = utils.removePadding(np.array(predicted), originalShape, valuesSplit)
                predicted = utils.applyMaskToVolume(predicted, mask_val)
                assert(predicted.shape == Y_val.shape)

                # Calculate metrics over validation and save it
                metrics = utils.getMetrics(Y_val, predicted, mask_val)
                summary = sess.run(val_merged_summaries, feed_dict={ rmseTensor: metrics[0], ddrmseTensor: metrics[1], is_train : False })
                val_summary_writer.add_summary(summary, globalStep)

                # Iterate over metrics
                print(metrics)
                for numMetric in range(len(metrics)):
                    
                    # If better value of metric, save it
                    if metrics[numMetric] < bestValMetrics[numMetric]:
                        bestValMetrics[numMetric] = metrics[numMetric]
                        saver.save(sess, checkpointsPath + "/model-metric" + str(numMetric))
            globalStep += 1 
        except tf.errors.OutOfRangeError:
            break
