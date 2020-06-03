import tensorflow as tf
import numpy as np

from utils import misc
from model import loss


class Solver():
    def __init__(self, params, ckp_path, modes, optimizer, training=True):
        self.params = params
        self.ckp_path = ckp_path
        self.modes = modes
        self.training = training

        # Class containing losses and metrics
        self.loss_manager = loss.LossManager()

        # Optimizer
        if self.training:
            self.generator_optimizer = optimizer(self.params["lr"], beta_1=0.5)
            self.discriminator_optimizer = optimizer(self.params["lr"], beta_1=0.5)

            misc.save_json(self.ckp_path + "params.json", self.params)

        # Best metrics
        self.best_metrics = {key: {"RMSE": 1000, "ddRMSE": 1000, "L1": 1000} for key in modes}
        self.consecutive_check_validation = 0

        # Tensorboard
        self.summary_writers, self.tb_scalars = self.init_tensorboard()

    def init_tensorboard(self):
        """ Initialize tensorboard variables. """
        summary_writers = {mode: tf.summary.create_file_writer(self.ckp_path + 'logs/' + mode) for mode in self.modes} if self.training else None
        tb_scalars = {"G_total": None, "G_loss": None, "D_real": None, "D_fake": None, "D_total": None}

        for metric in list(tb_scalars.keys()) + list(self.best_metrics[self.modes[0]].keys()):
            tb_scalars[metric] = tf.keras.metrics.Mean(metric, dtype=tf.float32)

        return summary_writers, tb_scalars

    def write_tensorboard(self, summary_writer, images, shape, epoch):
        """ Write stats in tensorboard file. """
        with summary_writer.as_default():
            for key in self.tb_scalars:
                tf.summary.scalar(key, self.tb_scalars[key].result(), step=epoch)

            # Tensorboard images
            for key in list(images.keys()):
                tf.summary.image(key, images[key][:, :, :, int(shape // 2)], max_outputs=1, step=epoch)

    def reset_states_tensorboard(self):
        """ After each epoch reset accumulating tensorboard metrics. """
        for key in self.tb_scalars:
            self.tb_scalars[key].reset_states()

    def iterate_dataset(self, G, D, dataset, epoch, mode):

        for batch in dataset:
            mask_batch = None

            if mode == "train" or mode == "val":
                x_batch, y_batch = batch[0], batch[1]

                # Crop if cropped shape different than initial one TODO : Crop randomly
                if np.any(self.params["crop_shape"] != y_batch.shape[1:4]):
                    y_batch = y_batch[:, :self.params["crop_shape"][0], :self.params["crop_shape"][1], :self.params["crop_shape"][2], :]
                    x_batch = x_batch[:, :self.params["crop_shape"][0], :self.params["crop_shape"][1], :self.params["crop_shape"][2], :]

            elif "qsm" in mode:
                mask_batch = batch["mask"]
                y_batch = misc.apply_mask(batch["y"], mask_batch)
                x_batch = misc.apply_mask(batch["x"], mask_batch)

            if mode == "train":
                predicted = self.train_step(G, D, x_batch, y_batch, mask_batch, mode)
            else:
                predicted = self.test_step(G, D, x_batch, y_batch, mask_batch, mode)
            break

        metrics = {key: self.tb_scalars[key].result().numpy() for key in self.best_metrics[mode]}
        print(metrics)
        self.write_tensorboard(self.summary_writers[mode], {"input": x_batch, "ground_truth": y_batch, "predicted": predicted}, predicted.shape[-2], epoch)
        self.reset_states_tensorboard()

        if mode != "train":
            self.save_model(G, mode, metrics)

    def train_step(self, G, D, x_batch, y_batch, mask_batch, mode):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = G(x_batch, training=True)

            disc_real_output = D(x_batch, y_batch, training=True)
            disc_generated_output = D(x_batch, gen_output, training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.loss_manager.generator_loss(self, disc_generated_output, gen_output, y_batch, self.params["lambda"])
            disc_loss = self.loss_manager.discriminator_loss(self, disc_real_output, disc_generated_output, self.params["label_smoothing"])

            generator_gradients = gen_tape.gradient(gen_total_loss, G.trainable_variables)
            discriminator_gradients = disc_tape.gradient(disc_loss, D.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(generator_gradients, G.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, D.trainable_variables))

            # Other metrics
            self.loss_manager.rmse_loss(self, y_batch, gen_output, mask_batch)

        return gen_output

    def test_step(self, G, D, x_batch, y_batch, mask_batch):

        gen_output = G(x_batch, training=True)

        disc_real_output = D(x_batch, y_batch, training=True)
        disc_generated_output = D(x_batch, gen_output, training=True)

        self.loss_manager.generator_loss(self, disc_generated_output, gen_output, y_batch, self.params["lambda"])
        self.loss_manager.discriminator_loss(self, disc_real_output, disc_generated_output, self.params["label_smoothing"])
        self.loss_manager.rmse_loss(self, y_batch, gen_output, mask_batch)

        return gen_output

    def save_model(self, model, mode, metrics):

        for key in metrics:

            better_metric = metrics[key] <= self.best_metrics[mode][key]
            if better_metric:
                self.consecutive_check_validation = 0
                self.best_metrics[mode][key] = metrics[key]
                model.save_weights(self.ckp_path + mode + "-" + key)

        self.consecutive_check_validation += 1
