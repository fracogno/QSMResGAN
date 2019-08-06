import tensorflow as tf
import numpy as np
import datetime
import os
import scipy.io
import network, utilities as util

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Paths
base_path = "/scratch/cai/QSM-GAN/"
data_path = "data/shapes_shape64_ex100_2018_10_18"

'''
    Parameters for training
'''
epochs = 1000
batch_size = 1
lr = 0.0002
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
'''VAL_SIZE = 256
X_val = scipy.io.loadmat("/scratch/cai/QSM-GAN/challenge/VALIDATION_forward.mat")["VALIDATION_forward"]
msk = scipy.io.loadmat(base_path + "challenge/msk.mat")["msk"]
X_val = X_val * msk
Y_val = scipy.io.loadmat(base_path + "challenge/chi_33.mat")["chi_33"]	
#X_val *= 10

val_X = (VAL_SIZE - X_val.shape[0]) // 2
val_Y = (VAL_SIZE - X_val.shape[1]) // 2
val_Z = (VAL_SIZE - X_val.shape[2]) // 2
X_val = np.pad(X_val, [(val_X, ), (val_Y, ), (val_Z, )],  'constant', constant_values=(0.0))
print(X_val.shape)
print(Y_val.shape)
X_val_tensor = tf.placeholder(tf.float32, shape=[None, VAL_SIZE, VAL_SIZE, VAL_SIZE, 1], name='X_val')'''

'''
    Define graphs for the networks
'''
Y_generated = network.getGenerator(X_tensor)    
#Y_val_generated = network.getGenerator(X_val_tensor)

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
D_loss_summary = tf.summary.scalar('D_loss', D_loss)
G_loss_summary = tf.summary.scalar('G_loss', G_gan)
L1_loss_summary = tf.summary.scalar('L1_loss', G_L1)

tf.summary.image('input', X_tensor[:, :, :, int(input_shape[2]/2)], max_outputs=1)
tf.summary.image('output', Y_generated[:, :, :, int(input_shape[2]/2)], max_outputs=1)
tf.summary.image('ground_truth', Y_tensor[:, :, :, int(input_shape[2]/2)], max_outputs=1)

# MERGE SUMMARIES - Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# UNMERGED SUMMARIES
avg_L1_loss = tf.placeholder(tf.float32, shape=())
val_L1_summary = tf.summary.scalar('val_L1_loss', avg_L1_loss)

# VISUALIZE => tensorboard --logdir=.
summaries_dir = base_path + checkpointName

with tf.Session() as sess:
    # Initialize variables
    best_val_L1 = 1e6
    saver = tf.train.Saver(max_to_keep=1)
    sess.run(tf.global_variables_initializer())
    
    # op to write logs to Tensorboard
    train_summary_writer = tf.summary.FileWriter(summaries_dir + '/train', graph=tf.get_default_graph())
    val_summary_writer = tf.summary.FileWriter(summaries_dir + '/val')

    global_step = 0
    while True:
        try:
            # Training step
            if global_step % 100 == 0:
                _, summary = sess.run([train_op, merged_summary_op])
                train_summary_writer.add_summary(summary, global_step)
            else:
                sess.run(train_op)
            
            # Check validation accuracy
            '''if global_step % 2500 == 0:
                Y_gen = sess.run(Y_val_generated, feed_dict={ X_val_tensor : [np.expand_dims(X_val, axis=-1)] })

                # Denormalize
                #Y_gen /= 10

                Y_gen = Y_gen[0, val_X:-val_X, val_Y:-val_Y, val_Z:-val_Z, 0]
                assert(Y_gen.shape[0] == msk.shape[0] and Y_gen.shape[1] == msk.shape[1] and Y_gen.shape[2] == msk.shape[2])

                # Calculate RMSE over validation and save it
                val_ddrmse = util.computeddRMSE(Y_val, Y_gen, msk)
                acc_summary = sess.run(val_L1_summary, feed_dict={ avg_L1_loss: val_ddrmse })
                val_summary_writer.add_summary(acc_summary, global_step)

                # If better accuracy, save it
                if val_ddrmse < best_val_L1:
                    best_val_L1 = val_ddrmse
                    save_path = saver.save(sess, summaries_dir + "/model")'''
            global_step += 1 
        except tf.errors.OutOfRangeError:
            break