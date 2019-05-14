import tensorflow as tf
import numpy as np
import datetime
import nibabel as nib

import Pix2Pix

base_path = "/home/francesco/UQ/Job/QSM-GAN/"

#last_path = "shapes_shape64_ex512_2019_05_01"
#last_path = "shapes_shape64_ex15_2019_04_17"
#last_path = "shapes_shape16_ex5_2019_03_26"
last_path = "shapes_shape16_ex128_2019_04_17"

currenttime = datetime.datetime.now()
checkpointName = "checkpoints_" + str(currenttime.year) + "-" + str(currenttime.month) + "-" + \
                    str(currenttime.day) + "_" + str(currenttime.hour) + str(currenttime.minute) + \
                    "_" + last_path

X, Y = [], []
for i in range(1, 129):
    X.append(nib.load(base_path + "data/" + last_path + "/traintrain1-size16-ex128_" + str(i) + "_forward_tfrange.nii").get_data())
    Y.append(nib.load(base_path + "data/" + last_path + "/traintrain1-size16-ex128_" + str(i) + "_ground_truth_tfrange.nii").get_data())

X = np.array(X)
Y = np.array(Y)

X = np.expand_dims(X, axis=-1)
Y = np.expand_dims(Y, axis=-1)

print(X.shape)
print(Y.shape)

X_tensor = tf.placeholder(tf.float32, shape=[None, X.shape[1], X.shape[2], X.shape[3], X.shape[4]], name='X')
Y_tensor = tf.placeholder(tf.float32, shape=[None, Y.shape[1], Y.shape[2], Y.shape[3], Y.shape[4]], name='Y')

# Create networks
with tf.variable_scope("generator"):
    Y_generated = Pix2Pix.getGenerator(X_tensor)

with tf.name_scope("real_discriminator"):
    with tf.variable_scope("discriminator"):
        D_logits_real = Pix2Pix.getDiscriminator(X_tensor, Y_tensor)
        
with tf.name_scope("fake_discriminator"):
    with tf.variable_scope("discriminator", reuse=True):
        D_logits_fake = Pix2Pix.getDiscriminator(X_tensor, Y_generated)


accuracy = tf.reduce_mean(tf.abs(Y_generated - Y_tensor))

# Parameters
lr = 0.0002
batch_size = 1
EPS = 1e-12

# Losses
def discriminatorLoss(D_real, D_fake):
   
    discrim_loss = tf.reduce_mean(-(tf.log(D_real + EPS) + tf.log(1 - D_fake + EPS)))
    
    return discrim_loss

def generatorLoss(D_fake, G_output, target, weight=100.0):
    gen_loss_GAN = tf.reduce_mean(-tf.log(D_fake + EPS))
    gen_loss_L1 = tf.reduce_mean(tf.abs(target -  G_output))
    gen_loss = gen_loss_GAN * 1.0 + gen_loss_L1 * weight

    return gen_loss, gen_loss_GAN, gen_loss_L1


D_loss = discriminatorLoss(D_logits_real, D_logits_fake)
G_loss, G_gan, G_L1 = generatorLoss(D_logits_fake, Y_generated, Y_tensor, 100.0)


D_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
D_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(D_loss, var_list=D_var)

G_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
G_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(G_loss, var_list=G_var)

'''
# Optimizers
with tf.name_scope('optimize'):
    discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
    discrim_optim = tf.train.AdamOptimizer(lr, 0.5)
    discrim_grads_and_vars = discrim_optim.compute_gradients(D_loss, var_list=discrim_tvars)
    discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.control_dependencies([discrim_train]):
        gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        gen_optim = tf.train.AdamOptimizer(lr, 0.5)
        gen_grads_and_vars = gen_optim.compute_gradients(G_loss, var_list=gen_tvars)
        gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

train_op = gen_train'''

# SUMMARIES - Create a list of summaries
tf.summary.image('input', X_tensor[:, :, :, int(X.shape[2]/2)], max_outputs=1)
tf.summary.image('output', Y_generated[:, :, :, int(X.shape[2]/2)], max_outputs=1)
tf.summary.image('ground_truth', Y_tensor[:, :, :, int(X.shape[2]/2)], max_outputs=1)

# MERGE SUMMARIES - Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# NON-MERGED SUMMARIES
avg_D_loss = tf.placeholder(tf.float32, shape=())
avg_G_loss = tf.placeholder(tf.float32, shape=())
avg_L1_loss = tf.placeholder(tf.float32, shape=())

D_loss_summary = tf.summary.scalar('D_loss', avg_D_loss)
G_loss_summary = tf.summary.scalar('G_loss', avg_G_loss)
L1_loss_summary = tf.summary.scalar('L1_loss', avg_L1_loss)


# VISUALIZE => tensorboard --logdir=.
summaries_dir = base_path + checkpointName

with tf.Session() as sess:
	# Initialize variables
	saver = tf.train.Saver(max_to_keep=1)
	sess.run(tf.global_variables_initializer())

	# op to write logs to Tensorboard
	train_summary_writer = tf.summary.FileWriter(summaries_dir + '/train', graph=tf.get_default_graph())
	val_summary_writer = tf.summary.FileWriter(summaries_dir + '/val')
	#val_avg_summary_writer = tf.summary.FileWriter(summaries_dir + '/val_avg')

	best_val_L1 = 100000
	for n_epoch in range(150):
		train_L1, train_G, train_D = [], [], []
		for j in range(0, len(X), batch_size):
			#_, D_loss_val, G_L1_val, G_gan_val = sess.run([train_op, D_loss, G_L1, G_gan],feed_dict={X_tensor:X[k:k+1],Y_tensor:Y[k:k+1]})
			
			# Optimize discriminato
			_, D_loss_val = sess.run([D_optimizer, D_loss], feed_dict={X_tensor: X[j:j+batch_size], Y_tensor: Y[j:j+batch_size]})

			# Optimizer generator
			_, G_loss_val = sess.run([G_optimizer, G_gan], feed_dict={X_tensor: X[j:j+batch_size],Y_tensor: Y[j:j+batch_size]})

			G_gan_val, D_loss_val, G_L1_val = sess.run([G_gan, D_loss, G_L1], 
				feed_dict={X_tensor: X[j:j+batch_size],
							Y_tensor: Y[j:j+batch_size]})


			print("Epoch " + str(n_epoch) + " " + str(D_loss_val) + " " + str(G_gan_val) + " " + str(G_L1_val))
			train_D.append(D_loss_val)
			train_L1.append(G_L1_val)
			train_G.append(G_gan_val)

		# Save averages of epoch to tensorboard
		feed_dict = { 
			avg_D_loss : np.mean(np.array(train_D)),
			avg_G_loss : np.mean(np.array(train_G)),
			avg_L1_loss : np.mean(np.array(train_L1))
		}
		L1_summary, G_summary, D_summary = sess.run([D_loss_summary, G_loss_summary, L1_loss_summary], feed_dict=feed_dict)
		train_summary_writer.add_summary(L1_summary, n_epoch)
		train_summary_writer.add_summary(D_summary, n_epoch)
		train_summary_writer.add_summary(G_summary, n_epoch)
