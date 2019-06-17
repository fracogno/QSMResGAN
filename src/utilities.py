import os
import tensorflow as tf
import numpy as np

def generate_file_list(file_path, p_shape):
    """
    Generates a list of the filenames and the paths in 'filepath'
    :param file_path: the path to the folder where the files of interest resides
    :param p_shape:
    :return: a list where the filenames and filepaths have been joined
    """
    filenames = os.listdir(file_path)

    for index, item in enumerate(filenames):
        if item.__contains__('size' + str(p_shape[0])):
            filenames[index] = file_path + item
        else:
            raise FileNotFoundError('you have files in the folder that does not match the shapes')

    return filenames


def data_input_fn(filenames, p_shape, batch=None, nepochs=None, shuffle=True):
    def _parser(record):
        features = {
            'forward_img': tf.FixedLenFeature([], tf.string, default_value=""),
            'ground_truth_img': tf.FixedLenFeature([], tf.string, default_value="")
        }
        parsed_record = tf.parse_single_example(record, features)
        forward_image = tf.decode_raw(parsed_record['forward_img'], tf.float32)
        forward_image = tf.reshape(forward_image, [p_shape[0], p_shape[1], p_shape[2], 1])

        ground_truth = tf.decode_raw(parsed_record['ground_truth_img'], tf.float32)
        ground_truth = tf.reshape(ground_truth, [p_shape[0], p_shape[1], p_shape[2], 1])

        return {"x": forward_image}, ground_truth

    def _input_fn():

        dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP').map(_parser)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.repeat(nepochs)
        dataset = dataset.batch(batch)
        # TODO some dataset cache can be made so you can cache on the harddisk

        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()

        return features, labels, dataset

    return _input_fn


def norm(data):
    """
    Normalizes the input by subtracting by the mean and dividing by the standard deviation
    :param data: input data to be normalized
    :param includemeanstd: Boolean whether to return the mean and standard deviation
    :return: Normalized data and mean and standard deviation
    """
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean) / std

    return data, mean, std