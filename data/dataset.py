import os

from functools import partial
from typing import Tuple

import tensorflow as tf

from utils.augmentation import aug_func
from utils.visualize import visualize_numpy_ds


def dataset_reader(path: str, extension='png', shuffle=0) -> tf.data.Dataset:
    file_paths = tf.io.gfile.glob(os.path.join(path, f'**/*.{extension}'))
    file_paths.sort()
    if shuffle > 0:
        file_paths = tf.random.shuffle(file_paths, shuffle)
    files_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    return files_ds


def decode_file(file: tf.Tensor, labels: list) -> Tuple[tf.Tensor, tf.Tensor]:
    image_binary = tf.io.read_file(file)
    image = tf.io.decode_jpeg(contents=image_binary)
    label = tf.strings.split(file, os.path.sep)[2]
    labels_encoded = tf.cast(label == labels, dtype=tf.int8)
    return image, labels_encoded


def get_label(train_path: str, test_path: str) -> list:
    train_labels = tf.io.gfile.listdir(train_path)
    train_labels.sort()
    test_labels = tf.io.gfile.listdir(test_path)
    test_labels.sort()
    assert train_labels == test_labels
    return train_labels


class Dataset:
    def __init__(self, path: str, labels: list, batch_size: int, is_training: bool, random_seed=0):
        self.path = path
        self.batch_size = batch_size
        self.is_training = is_training
        self.random_seed = random_seed
        self.labels = labels
        self.autotune = tf.data.experimental.AUTOTUNE
        self.files = dataset_reader(self.path, shuffle=self.random_seed)
        self.ds = self.create_dataset(self.files)
        self.configure_dataset()

    def create_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        decoded_ds = dataset.map(partial(decode_file, labels=self.labels), num_parallel_calls=self.autotune)
        processed_ds = decoded_ds.map(partial(aug_func, is_training=self.is_training), self.autotune)
        return processed_ds

    def configure_dataset(self):
        self.ds = self.ds.batch(self.batch_size).cache().prefetch(self.autotune)


def debug_dataset(train_path='data/train', test_path='data/eval'):
    classes = get_label(train_path, test_path)
    dataset = Dataset('data/train', labels=classes, batch_size=32, is_training=True, random_seed=20)
    for img, label in dataset.ds.unbatch().take(100).as_numpy_iterator():
        if visualize_numpy_ds(img, label, classes):
            continue
        else:
            break


if __name__ == '__main__':
    debug_dataset()
