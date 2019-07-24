import tensorflow as tf
import numpy as np
import nibabel as nib
import datetime
import matplotlib.pyplot as plt
import os

import Pix2Pix, utilities as util

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Paths
base_path = "/scratch/cai/QSM-GAN/"
data_path = "data/shapes_shape64_ex100_2018_10_18"
#data_path = "data/TMPDATA"

input_shape = (64, 64, 64, 1)

'''
    Parameters for training
'''
epochs = 1000
batch_size = 1
lr = 0.0002
beta1 = 0.5
l1_weight = 100.0

# Create checkpoints path
tf.reset_default_graph()
now = datetime.datetime.now()
checkpointName = "ckp_" + str(now.year) + str(now.month) + str(now.day) + "_" + str(now.hour) + str(now.minute) + "_" + data_path.split("/")[-1]

'''
    Import data
'''
# Training
train_data_filename = util.generate_file_list(file_path=base_path + data_path + "/train/", p_shape=input_shape)
train_input_fn = util.data_input_fn(train_data_filename, p_shape=input_shape, batch=batch_size, nepochs=epochs, shuffle=True)
train_data = train_input_fn()
iterator = tf.data.Iterator.from_structure(train_data[2].output_types, train_data[2].output_shapes)
X_tensor, Y_tensor = iterator.get_next()
training_init_op = iterator.make_initializer(train_data[2])

# Validation
VAL_SIZE = 256
X_val = nib.load("/scratch/cai/QSM-GAN/challenge/phs_tissue.nii").get_data()
Y_val = nib.load("/scratch/cai/QSM-GAN/challenge/chi_33.nii").get_data()
msk = nib.load("/scratch/cai/QSM-GAN/challenge/msk.nii").get_data()
X_val, phase_mean, phase_std = util.norm(X_val)

val_X = (VAL_SIZE - X_val.shape[0]) // 2
val_Y = (VAL_SIZE - X_val.shape[1]) // 2
val_Z = (VAL_SIZE - X_val.shape[2]) // 2
X_val = np.pad(X_val, [(val_X, ), (val_Y, ), (val_Z, )],  'constant', constant_values=(0.0))
print(X_val.shape)
print(Y_val.shape)
X_val_tensor = tf.placeholder(tf.float32, shape=[None, VAL_SIZE, VAL_SIZE, VAL_SIZE, 1], name='X')

'''
    Define graphs for the networks
'''
with tf.variable_scope("generator"):
    Y_generated = Pix2Pix.getGenerator(X_tensor)

with tf.variable_scope("generator", reuse=True):
    Y_val_generated = Pix2Pix.getGenerator(X_val_tensor)

with tf.variable_scope("discriminator"):
    D_logits_real = Pix2Pix.getDiscriminator(X_tensor, Y_tensor)
        
with tf.variable_scope("discriminator", reuse=True):
    D_logits_fake = Pix2Pix.getDiscriminator(X_tensor, Y_generated)

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

D_loss = discriminatorLoss(D_logits_real, D_logits_fake, 0.9)
G_loss, G_gan, G_L1 = generatorLoss(D_logits_fake, Y_generated, Y_tensor, l1_weight)

'''
    Optimizers for weights
'''
with tf.name_scope('optimize'):
    discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
    discrim_optim = tf.train.AdamOptimizer(lr, beta1)
    discrim_grads_and_vars = discrim_optim.compute_gradients(D_loss, var_list=discrim_tvars)
    discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.control_dependencies([discrim_train]):
        gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        gen_optim = tf.train.AdamOptimizer(lr, beta1)
        gen_grads_and_vars = gen_optim.compute_gradients(G_loss, var_list=gen_tvars)
        gen_train = gen_optim.apply_gradients(gen_grads_and_vars)
train_op = gen_train

'''
    SUMMARIES - Create a list of summaries
'''
D_loss_summary = tf.summary.scalar('D_loss', D_loss)
G_loss_summary = tf.summary.scalar('G_loss', G_loss)
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

    sess.run(training_init_op)
    global_step = 0
    while True:
        try:
            # Training step
            if global_step % 2000 == 0:
                _, summary = sess.run([train_op, merged_summary_op])
                train_summary_writer.add_summary(summary, global_step)
            else:
                sess.run(train_op)
            
            # Check validation accuracy
            if global_step % 2000 == 0:
                Y_gen = sess.run(Y_val_generated, feed_dict={ X_val_tensor : [np.expand_dims(X_val, axis=-1)] })

                # Denormalize
                Y_gen = Y_gen + (3 * phase_mean)
                Y_gen = Y_gen * (3 * phase_std)

                Y_gen = Y_gen[0, val_X:-val_X, val_Y:-val_Y, val_Z:-val_Z, 0]
                assert(Y_gen.shape[0] == msk.shape[0] and Y_gen.shape[1] == msk.shape[1] and Y_gen.shape[2] == msk.shape[2])

                # Calculate RMSE over validation and save it
                val_ddrmse = util.computeddRMSE(Y_val, Y_gen, msk):
                acc_summary = sess.run(val_L1_summary, feed_dict={ avg_L1_loss: val_ddrmse })
                val_summary_writer.add_summary(acc_summary, global_step)

                # If better accuracy, save it
                if val_ddrmse < best_val_L1:
                    best_val_L1 = val_ddrmse
                    save_path = saver.save(sess, summaries_dir + "/model")
            global_step += 1 
        except tf.errors.OutOfRangeError:
            break