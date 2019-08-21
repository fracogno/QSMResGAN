import tensorflow as tf
import numpy as np
import datetime
import os
import nibabel as nib
import src.network as network, src.utilities as util

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Paths
base_path = "/scratch/cai/deepQSMGAN/"
base_path = "/home/francesco/UQ/deepQSMGAN/"
data_path = "data/shapes_shape64_ex2_2019_08_19"

'''
    Parameters for training
'''
epochs = 3
batch_size = 1
lr = 0.0001
beta1 = 0.5
l1_weight = 100.0
labelSmoothing = 0.9

# Create checkpoints path
tf.reset_default_graph()
now = datetime.datetime.now()
checkpointName = "ckp_" + str(now.year) + str(now.month) + str(now.day) + "_" + str(now.hour) + str(now.minute) + "_" + data_path.split("/")[-1]

'''
    Import data
'''
# Training
input_shape = (64, 64, 64, 1)
train_data_filename = util.generate_file_list(file_path=base_path + data_path + "/train/", p_shape=input_shape)
train_input_fn = util.data_input_fn(train_data_filename, p_shape=input_shape, batch=batch_size, nepochs=epochs, shuffle=True)
X_tensor, Y_tensor, _ = train_input_fn()

# Validation
X_val_nib = nib.load(base_path + "QSM_Challenge2_download_stage2/DatasetsStep2/Sim2Snr2/Frequency.nii.gz")
X_tmp = X_val_nib.get_data()
Y_tmp = nib.load(base_path + "QSM_Challenge2_download_stage2/DatasetsStep2/Sim2Snr2/GT/Chi.nii.gz")
Y_val = Y_tmp.get_data()
mask = nib.load(base_path + "QSM_Challenge2_download_stage2/DatasetsStep2/Sim2Snr2/MaskBrainExtracted.nii.gz").get_data()
finalSegment = nib.load(base_path + "QSM_Challenge2_download_stage2/DatasetsStep2/Sim2Snr2/GT/Segmentation.nii.gz").get_data()

# Rescale validation
TEin_s = 8 / 1000
frequency_rad = X_tmp * TEin_s * 2 * np.pi
centre_freq = 297190802
X_val_original = frequency_rad / (2 * np.pi * TEin_s * centre_freq) * 1e6
print("X val original shape " + str(X_val_original.shape))

# Add one if shape is not EVEN
X_val = np.pad(X_val_original, [(int(X_val_original.shape[0] % 2 != 0), 0), (int(X_val_original.shape[1] % 2 != 0), 0), (int(X_val_original.shape[2] % 2 != 0), 0)],  'constant', constant_values=(0.0))
print("X val evened shape " + str(X_val.shape))

# Pad to multiple of 2^n
VAL_SIZE = 256
X_val_tensor = tf.placeholder(tf.float32, shape=[None, VAL_SIZE, VAL_SIZE, VAL_SIZE, 1], name='X_val')
val_X = (VAL_SIZE - X_val.shape[0]) // 2
val_Y = (VAL_SIZE - X_val.shape[1]) // 2
val_Z = (VAL_SIZE - X_val.shape[2]) // 2
X_val = np.pad(X_val, [(val_X, ), (val_Y, ), (val_Z, )],  'constant', constant_values=(0.0))
print("X val padded to 2^n multiple: " + str(X_val.shape))
print("Y val: " + str(Y_val.shape))
print("Mask shape: " + str(mask.shape))

'''
    Define graphs for the networks
'''
Y_generated = network.getGenerator(X_tensor)    
Y_val_generated = network.getGenerator(X_val_tensor, True)

D_logits_real = network.getDiscriminator(X_tensor, Y_tensor)
D_logits_fake = network.getDiscriminator(X_tensor, Y_generated)

'''
    Loss functions
'''
def discriminatorLoss(dis_real, dis_fake, smoothing=1.0):
    dis_real_ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_real, labels=tf.ones_like(dis_real) * smoothing)
    dis_fake_ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake, labels=tf.zeros_like(dis_fake))

    dis_loss_real = tf.reduce_mean(dis_real_ce)
    dis_loss_fake = tf.reduce_mean(dis_fake_ce)
    dis_loss = tf.reduce_mean(dis_real_ce + dis_fake_ce)

    return dis_loss

def generatorLoss(dis_fake, G_output, target, weight):
    gen_ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake, labels=tf.ones_like(dis_fake))

    gen_loss_gan = tf.reduce_mean(gen_ce)
    gen_loss_l1 = tf.reduce_mean(tf.abs(target - G_output)) 
    gen_loss = gen_loss_gan + (gen_loss_l1 * weight)

    return gen_loss, gen_loss_gan, gen_loss_l1

D_loss = discriminatorLoss(D_logits_real, D_logits_fake, labelSmoothing)
G_loss, G_gan, G_L1 = generatorLoss(D_logits_fake, Y_generated, Y_tensor, l1_weight)

'''
    Optimizers for weights
'''
D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')
with tf.control_dependencies(D_update_ops):
	D_optimizer = tf.train.AdamOptimizer(lr, beta1).minimize(D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))

	with tf.control_dependencies([D_optimizer]):
		G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')

		with tf.control_dependencies(G_update_ops):
			G_optimizer = tf.train.AdamOptimizer(lr, beta1).minimize(G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))
train_op = G_optimizer

'''
    SUMMARIES - Create a list of summaries
'''
# Training
train_summaries = [tf.summary.scalar('D_loss', D_loss), tf.summary.scalar('G_loss', G_gan), tf.summary.scalar('L1_loss', G_L1), \
					tf.summary.image('input', X_tensor[:, :, :, int(input_shape[2]/2)], max_outputs=1), \
					tf.summary.image('output', Y_generated[:, :, :, int(input_shape[2]/2)], max_outputs=1), \
					tf.summary.image('ground_truth', Y_tensor[:, :, :, int(input_shape[2]/2)], max_outputs=1)]

train_merged_summaries = tf.summary.merge(train_summaries)

# Validation
rmseTensor = tf.placeholder(tf.float32, shape=())
ddrmseTensor = tf.placeholder(tf.float32, shape=())
ddrmseTissueTensor = tf.placeholder(tf.float32, shape=())
ddrmseBloodTensor = tf.placeholder(tf.float32, shape=())
ddrmseDGMTensor = tf.placeholder(tf.float32, shape=())
deviationFromLinearSlopeTensor = tf.placeholder(tf.float32, shape=())
calcStreakTensor = tf.placeholder(tf.float32, shape=())
deviationFromCalcMomentTensor = tf.placeholder(tf.float32, shape=())

val_summaries = [tf.summary.scalar('rmse', rmseTensor), tf.summary.scalar('ddrmse', ddrmseTensor), tf.summary.scalar('ddrmseTissue', ddrmseTissueTensor), \
				 tf.summary.scalar('ddrmseBlood', ddrmseBloodTensor), tf.summary.scalar('ddrmseDGM', ddrmseDGMTensor), \
				 tf.summary.scalar('deviationFromLinearSlope', deviationFromLinearSlopeTensor), tf.summary.scalar('calcStreak', calcStreakTensor), \
				 tf.summary.scalar('deviationFromCalcMoment', deviationFromCalcMomentTensor)]
val_merged_summaries = tf.summary.merge(val_summaries)

# VISUALIZE => tensorboard --logdir=.
summaries_dir = base_path + checkpointName

# Array containing best metrics values
bestValMetrics = [1e6] * len(util.getMetrics(Y_val, Y_val, mask, finalSegment))

with tf.Session() as sess:
    # Initialize variables
    saver = tf.train.Saver(max_to_keep=15)
    sess.run(tf.global_variables_initializer())

    # op to write logs to Tensorboard
    train_summary_writer = tf.summary.FileWriter(summaries_dir + '/train', graph=tf.get_default_graph())
    val_summary_writer = tf.summary.FileWriter(summaries_dir + '/val')

    global_step = 0
    while True:
        try:
            # Training step
            if global_step % 1 == 0:
                _, summary = sess.run([train_op, train_merged_summaries])
                train_summary_writer.add_summary(summary, global_step)
            else:
                sess.run(train_op)
            
            # Check validation accuracy
            if global_step % 1 == 0:
                predicted = sess.run(Y_val_generated, feed_dict={ X_val_tensor : [np.expand_dims(X_val, axis=-1)] })
                
                #Remove paddings and if it was not even shape
                predicted = predicted[0, val_X:-val_X, val_Y:-val_Y, val_Z:-val_Z, 0]
                predicted = predicted[int(X_val_original.shape[0] % 2 != 0):, int(X_val_original.shape[1] % 2 != 0):, int(X_val_original.shape[2] % 2 != 0):]
                assert(predicted.shape[0] == mask.shape[0] and predicted.shape[1] == mask.shape[1] and predicted.shape[2] == mask.shape[2])
                predicted = predicted * mask

                # Calculate metrics over validation and save it
                metrics = util.getMetrics(Y_val, predicted, mask, finalSegment)
                summary = sess.run(val_merged_summaries, feed_dict={ rmseTensor: metrics[0],
                													 ddrmseTensor: metrics[1],
                													 ddrmseTissueTensor: metrics[2],
                													 ddrmseBloodTensor: metrics[3],
                													 ddrmseDGMTensor: metrics[4],
                													 deviationFromLinearSlopeTensor: metrics[5],
                													 calcStreakTensor: metrics[6],
                													 deviationFromCalcMomentTensor: metrics[7] })
                val_summary_writer.add_summary(summary, global_step)

                # Iterate over metrics
                print(metrics)
                for numMetric in range(len(metrics)):
                    
                    # If better value of metric, save it
                    if metrics[numMetric] < bestValMetrics[numMetric]:
                        bestValMetrics[numMetric] = metrics[numMetric]
                        saver.save(sess, summaries_dir + "/model-metric" + str(numMetric))
            global_step += 1 
        except tf.errors.OutOfRangeError:
            break