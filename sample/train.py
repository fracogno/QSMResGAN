import tensorflow as tf

from utils import misc, data_manager
from model import generator, discriminator, solver


def train(params):
    # Set up directory paths
    base_path, ckp_path = misc.get_base_path(training=True)

    # Get optimizer
    optimizer = params["optimizer"]
    params["optimizer"] = str(params["optimizer"])
    params["crop_shape"] = [64, 64, 64]

    dataset = data_manager.get_QSM_datasets(base_path + "dataset/", "shapes_shape64_ex100_2019_08_10", "20170327_qsm2016_recon_challenge",
                                            "QSM_Challenge2_download_stage2", 64, params["batch_size"], get_train_data=True)

    # Networks
    gen = generator.Generator(params, params["initializer"])
    disc = discriminator.Discriminator(params, params["initializer"])

    # Solver to train network
    slv = solver.Solver(params, ckp_path, list(dataset.keys()), optimizer)

    epoch = 0
    while True:
        # Iterate over train, validation, test datasets
        for mode in slv.modes:
            slv.iterate_dataset(gen, disc, dataset[mode], epoch, mode)

        if slv.consecutive_check_validation >= params["early_stopping"]:
            break
        epoch += 1

    # Save stats after training
    slv.best_metrics["epoch"] = epoch
    slv.best_metrics = {key: str(slv.best_metrics[key]) for key in slv.best_metrics}
    misc.save_json(slv.ckp_path + "training.json", slv.best_metrics)


if __name__ == "__main__":
    lr_vector = [1e-4]
    batch_size_vector = [2]
    kernel_size_vector = [3]
    optimizer_vector = [tf.keras.optimizers.Adam]  # Adam, Adamax, Nadam, Ftrl, RMSprop
    dropout_rate_vector = [0.]
    use_batch_norm_vector = [False]
    use_bias_vector = [False]
    lambda_vector = [100.]
    initializer_vector = ["he_normal"]  # tf.random_normal_initializer(0., 0.02)
    label_smoothing_vector = [0.9]
    early_stopping = 30

    for lr in lr_vector:
        for batch_size in batch_size_vector:
            for kernel_size in kernel_size_vector:
                for optimizer in optimizer_vector:
                    for dropout_rate in dropout_rate_vector:
                        for use_batch_norm in use_batch_norm_vector:
                            for label_smoothing in label_smoothing_vector:
                                for lambda_ in lambda_vector:
                                    for use_bias in use_bias_vector:
                                        for initializer in initializer_vector:
                                            train({"lr": lr,
                                                   "batch_size": batch_size,
                                                   "k_size": kernel_size,
                                                   "optimizer": optimizer,
                                                   "dropout_rate": dropout_rate,
                                                   "use_batch_norm": use_batch_norm,
                                                   "use_bias": use_bias,
                                                   "initializer": initializer,
                                                   "lambda": lambda_,
                                                   "early_stopping": early_stopping,
                                                   "label_smoothing": label_smoothing})
