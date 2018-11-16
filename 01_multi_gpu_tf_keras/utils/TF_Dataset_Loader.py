import tensorflow as tf
from PIL import Image
import numpy as np
import os


class TFDatasetLoader:
    def __init__(self, data_dirpath, input_size, n_classes, batch_size, epochs):
        self.train_files = self.decode_filepaths(os.path.join(data_dirpath, "train.txt"))
        self.val_files = self.decode_filepaths(os.path.join(data_dirpath, "val.txt"))
        self.input_size = [input_size, input_size, 3]
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.data_dirpath = data_dirpath
        
    def _one_hot_gt(self, image, label):
        gt_decoded = np.array(Image.open(label.decode()))
        label_tensor = (np.arange(2) == gt_decoded[:, :, None]).astype(np.uint8)
        return image, label_tensor

    def _data_to_tensor(self, image, label):
        image_string = tf.read_file(image)
        image_tensor = tf.image.convert_image_dtype(
            tf.image.decode_jpeg(image_string, channels=3), dtype=tf.float32
        )
        return image_tensor, label

    def imgs_input_fn(self, mode=None):
        image_paths = []
        gt_paths = []

        if mode=="train":
            for line in self.train_files:
                image_paths.append(os.path.join(self.data_dirpath, line.split(' ')[0]))
                gt_paths.append(os.path.join(self.data_dirpath, line.split(' ')[1]))
        else:
            for line in self.val_files:
                image_paths.append(os.path.join(self.data_dirpath, line.split(' ')[0]))
                gt_paths.append(os.path.join(self.data_dirpath, line.split(' ')[1]))

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, gt_paths))
        dataset = dataset.map(
            lambda image, label: tuple(tf.py_func(
                self._one_hot_gt, [image, label], [tf.string, tf.uint8])))
        dataset = dataset.map(self._data_to_tensor)
        dataset = dataset.repeat(self.epochs)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.batch_size)
        return dataset


    def decode_filepaths(self, data_dirpath):
        filepaths = []
        for line in open(data_dirpath, 'r').readlines():
            line = line.strip()
            if len(line) > 0:
                filepaths.append(line)
        return filepaths