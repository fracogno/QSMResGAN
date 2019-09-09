import tensorflow as tf


def discriminatorLoss(dis_real, dis_fake, smoothing):
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


def getOptimizer(lr, beta1, D_loss, G_loss):
    D_optimizer = tf.train.AdamOptimizer(lr, beta1).minimize(D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))
    with tf.control_dependencies([D_optimizer]):
        G_optimizer = tf.train.AdamOptimizer(lr, beta1).minimize(G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))
    return G_optimizer