import tensorflow as tf
import numpy as np


class LossManager:

    def __init__(self):
        self.ce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


    def wgan_generator_loss(self, solver, D_fake_logits):
        G_loss = - tf.reduce_mean(D_fake_logits)
        solver.tb_scalars["G_loss"](G_loss) if not solver is None else None
        return G_loss

    def wgan_discriminator_loss(self, solver, D, D_fake_logits, D_real_logits, input_gen, real_sample, fake_sample, LAMBDA=10.):
        # Gradient penalty
        eps = tf.random_uniform([D_fake_logits.shape[0], 1, 1, 1, 1], minval=0., maxval=1.)
        inter_sample = real_sample * eps + fake_sample * (1. - eps)
        with tf.GradientTape() as tape_gp:
            tape_gp.watch(inter_sample)
            inter_score = D(input_gen, inter_sample, training=True)

        gp_gradients = tape_gp.gradient(inter_score, inter_sample)
        gp_gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gp_gradients), axis=[1, 2, 3, 4]))
        gp = tf.reduce_mean(tf.square((gp_gradients_norm - 1.0)))

        # Loss
        D_loss = tf.reduce_mean(D_fake_logits) - tf.reduce_mean(D_real_logits) + gp * LAMBDA
        solver.tb_scalars["D_loss"](D_loss) if not solver is None else None

        return D_loss

    def generator_loss(self, solver, disc_generated_output, gen_output, target, LAMBDA):
        G_loss = self.ce(tf.ones_like(disc_generated_output), disc_generated_output)
        solver.tb_scalars["G_loss"](G_loss) if not solver is None else None

        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        solver.tb_scalars["L1"](l1_loss) if not solver is None else None

        total_gen_loss = G_loss + (LAMBDA * l1_loss)
        solver.tb_scalars["G_total"](total_gen_loss) if not solver is None else None

        return total_gen_loss, G_loss, l1_loss

    def discriminator_loss(self, solver, disc_real_output, disc_generated_output, smoothing):
        real_loss = self.ce(tf.ones_like(disc_real_output) * smoothing, disc_real_output)
        solver.tb_scalars["D_real"](real_loss) if not solver is None else None

        generated_loss = self.ce(tf.zeros_like(disc_generated_output), disc_generated_output)
        solver.tb_scalars["D_fake"](generated_loss) if not solver is None else None

        total_disc_loss = real_loss + generated_loss
        solver.tb_scalars["D_total"](total_disc_loss) if not solver is None else None

        return total_disc_loss

    def rmse_loss(self, solver, true, fake, mask):

        if mask is None:
            mask = np.ones_like(true)

        if not isinstance(true, np.ndarray):
            true = true.numpy()

        if not isinstance(fake, np.ndarray):
            fake = fake.numpy()

        if not isinstance(mask, np.ndarray):
            mask = mask.numpy()

        true_flat = true.flatten()
        fake_flat = fake.flatten()
        mask_flat = np.array(mask.flatten(), dtype=bool)

        # Get only elements in mask
        true_new = true_flat[mask_flat]
        fake_new = fake_flat[mask_flat]

        # Demean
        true_demean = true_new - np.mean(true_new);
        fake_demean = fake_new - np.mean(fake_new);

        rmse = 100 * np.linalg.norm(true_demean - fake_demean) / np.linalg.norm(true_demean);

        # Detrend
        P1 = np.polyfit(true_demean, fake_demean, 1)
        P = [0, 0]
        P[0] = 1 / P1[0]
        P[1] = -P1[1] / P1[0]
        res = np.polyval(P, fake_demean)

        # RMSE
        ddrmse = 100 * np.linalg.norm(res - true_demean) / np.linalg.norm(true_demean);

        solver.tb_scalars["RMSE"](rmse) if not solver is None else None  # Only save in tensorboard if solver not None
        solver.tb_scalars["ddRMSE"](ddrmse) if not solver is None else None  # Only save in tensorboard if solver not None

        return rmse, ddrmse
