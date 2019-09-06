import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import datetime
import src.ResUNET as network, src.utilities as util

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

lr = 1e-4
beta1 = 0.5

# Paths
base_path = "/scratch/cai/deepQSMGAN/"
data_path = "data/shapes_shape64_ex100_2019_08_30"

base_path ="/home/francesco/UQ/deepQSMGAN/"
data_path = "data/shapes_shape64_ex100_2019_08_20"

now = datetime.datetime.now()
checkpointName = "ckp_" + str(now.year) + str(now.month) + str(now.day) + "_" + str(now.hour) + str(now.minute) + "_" + data_path.split("/")[-1]

input_shape = (64, 64, 64, 1)
train_data_filename = util.generate_file_list(file_path=base_path + data_path + "/train/", p_shape=input_shape)
train_input_fn = util.data_input_fn(train_data_filename, p_shape=input_shape, batch=1, nepochs=3000, shuffle=True)
X_tensor, Y_tensor, _ = train_input_fn()


Y_generated = network.getGenerator(X_tensor)    
D_logits_real = network.getDiscriminator(X_tensor, Y_tensor)
D_logits_fake = network.getDiscriminator(X_tensor, Y_generated, True)


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

D_loss = discriminatorLoss(D_logits_real, D_logits_fake)
G_loss, G_gan, G_L1 = generatorLoss(D_logits_fake, Y_generated, Y_tensor, 100.0)

D_optimizer = tf.train.AdamOptimizer(lr, beta1).minimize(D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))
with tf.control_dependencies([D_optimizer]):
    G_optimizer = tf.train.AdamOptimizer(lr, beta1).minimize(G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))
optimizer = G_optimizer

train_summaries = [tf.summary.scalar('D_loss', D_loss), tf.summary.scalar('G_gan', G_gan), tf.summary.scalar('L1_loss', G_L1)]
train_merged_summaries = tf.summary.merge(train_summaries)

for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'):
    print(i)

for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'):
    print(i)

# VISUALIZE => tensorboard --logdir=.
summaries_dir = base_path + checkpointName

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    train_summary_writer = tf.summary.FileWriter(summaries_dir + '/train', graph=tf.get_default_graph())
    global_step = 0
    while True:
        try:
            _, summary = sess.run([optimizer, train_merged_summaries])
            train_summary_writer.add_summary(summary, global_step)
            global_step += 1 
        except tf.errors.OutOfRangeError:
            break
