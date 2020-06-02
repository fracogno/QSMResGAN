import tensorflow as tf
import os


class TFRecordManager:

    def __init__(self):
        pass

    def save_record(self, x, y, path):
        writer = tf.io.TFRecordWriter(path, options=tf.io.TFRecordOptions(compression_type="GZIP"))

        for index in range(len(x)):
            forward_string = x[index].tostring()
            ground_truth_string = y[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': self._int64_feature(x.shape[1]),
                'width': self._int64_feature(x.shape[2]),
                'depth': self._int64_feature(1),
                'ground_truth_img': self._bytes_feature(ground_truth_string),
                'forward_img': self._bytes_feature(forward_string)}))
            writer.write(example.SerializeToString())
        writer.close()

    def load_dataset(self, path, shape, batch_size, shuffle=True):
        dataset = tf.data.TFRecordDataset(self.get_records_filenames(path, shape), compression_type='GZIP').map(lambda record: self.parser_TFRecord(record, shape))

        if shuffle:
            dataset = dataset.shuffle(5000)
        dataset = dataset.batch(batch_size)

        return dataset

    def get_records_filenames(self, path, shape):
        filenames = os.listdir(path)
        for index in range(len(filenames)):

            # Check if input shape is correct
            if not 'size' + str(shape) in filenames[index]:
                raise FileNotFoundError('The current folder contains files whose shape do not match the input one.')
            filenames[index] = path + filenames[index]

        return filenames

    def parser_TFRecord(self, record, shape):
        features = {
            'forward_img': tf.io.FixedLenFeature([], tf.string, default_value=""),
            'ground_truth_img': tf.io.FixedLenFeature([], tf.string, default_value="")
        }
        parsed_record = tf.io.parse_single_example(record, features)
        forward_image = tf.io.decode_raw(parsed_record['forward_img'], tf.float32)
        forward_image = tf.reshape(forward_image, [shape, shape, shape, 1])

        ground_truth = tf.io.decode_raw(parsed_record['ground_truth_img'], tf.float32)
        ground_truth = tf.reshape(ground_truth, [shape, shape, shape, 1])

        return forward_image, ground_truth

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
