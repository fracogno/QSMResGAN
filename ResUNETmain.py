import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import datetime
import nibabel as nib
import src.ResUNET as ResUNET, src.utilities as util

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Paths
base_path = "/scratch/cai/deepQSMGAN/"
data_path = "data/shapes_shape64_ex100_2019_08_30"

#base_path = "/home/francesco/UQ/deepQSMGAN/"
#data_path = "data/shapes_shape64_ex100_2019_08_20"

# Parameters for training
epochs = 2500
batch_size = 1
lr = 0.0001
beta1 = 0.5
l1_weight = 100.0
labelSmoothing = 0.9

# Create checkpoints path
tf.reset_default_graph()
now = datetime.datetime.now()
checkpointName = "ckp_" + str(now.year) + str(now.month) + str(now.day) + "_" + str(now.hour) + str(now.minute) + "_" + data_path.split("/")[-1]

# Training data
input_shape = (64, 64, 64, 1)
train_data_filename = util.generate_file_list(file_path=base_path + data_path + "/train/", p_shape=input_shape)
train_input_fn = util.data_input_fn(train_data_filename, p_shape=input_shape, batch=batch_size, nepochs=epochs, shuffle=True)
X_tensor, Y_tensor, _ = train_input_fn()

# Define graphs for the networks
Y_generated = ResUNET.getGenerator(X_tensor)    

D_logits_real = ResUNET.getDiscriminator(X_tensor, Y_tensor)
D_logits_fake = ResUNET.getDiscriminator(X_tensor, Y_generated, True)

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

# Optimizers for weights
D_optimizer = tf.train.AdamOptimizer(lr, beta1).minimize(D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))
with tf.control_dependencies([D_optimizer]):
    G_optimizer = tf.train.AdamOptimizer(lr, beta1).minimize(G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))
train_op = G_optimizer

# SUMMARIES - Create a list of summaries
train_summaries = [tf.summary.scalar('D_loss', D_loss), tf.summary.scalar('G_loss', G_gan), tf.summary.scalar('L1_loss', G_L1), \
					tf.summary.image('input', X_tensor[:, :, :, int(input_shape[2]/2)], max_outputs=1), \
					tf.summary.image('output', Y_generated[:, :, :, int(input_shape[2]/2)], max_outputs=1), \
					tf.summary.image('ground_truth', Y_tensor[:, :, :, int(input_shape[2]/2)], max_outputs=1)]
train_merged_summaries = tf.summary.merge(train_summaries)

# VISUALIZE => tensorboard --logdir=.
summaries_dir = base_path + checkpointName

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # Initialize variables
    saver = tf.train.Saver(max_to_keep=5000)
    sess.run(tf.global_variables_initializer())

    # op to write logs to Tensorboard
    train_summary_writer = tf.summary.FileWriter(summaries_dir + '/train', graph=tf.get_default_graph())

    global_step = 0
    while True:
        try:
            # Training step
            if global_step % 50 == 0:
                _, summary = sess.run([train_op, train_merged_summaries])
                train_summary_writer.add_summary(summary, global_step)
            else:
                sess.run(train_op)

            if global_step % 1000 == 0:
                saver.save(sess, summaries_dir + "/model-step-" + str(global_step))
            global_step += 1 
        except tf.errors.OutOfRangeError:
            break
