import os

import numpy as np
import tensorflow as tf
import pandas as pd


class Dataset:
    def __init__(self, train_path, batch_size):
        self.train_path = train_path
        self.batch_size = batch_size

        self.autotune = tf.data.experimental.AUTOTUNE

        self.train_files = self.dataset_reader(self.train_path)
        self.train_ds = self.create_dataset(self.train_files)

    def create_dataset(self, files):
        files_ds = tf.data.Dataset.from_tensor_slices(files)
        image_label_ds = files_ds.map(self.path_decoder, num_parallel_calls=self.autotune)
        return image_label_ds

    @staticmethod
    def dataset_reader(path):
        file_paths = tf.io.gfile.glob(os.path.join(path, '**/*.jpeg'))
        return file_paths

    @staticmethod
    def path_decoder(file):
        image_binary = tf.io.read_file(file)
        image = tf.io.decode_jpeg(contents=image_binary)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        labels = tf.strings.split(file, os.path.sep)[2]
        return image, labels



if __name__ == '__main__':
    ds = Dataset('data/train', 32)
    print(np.array(list(ds.train_ds.take(10).as_numpy_iterator()))[0])
