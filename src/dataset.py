import numpy as np
import pandas as pd
import tensorflow as tf

# Create tf.data.Dataset datasets
def create_labels_dataset(labels: pd.DataFrame) -> tf.data.Dataset:
    """
    Create tf.data.Dataset datasets from labels in pd.DataFrame.
    During the process, it also normalizes the 'translation_vector' feature.

    :param labels: columns('quaternions', 'translation_vector', 'outliers')
    :return: tf.data.Dataset: (y_quaternions, y_transl_vect, y_outliers)
    """

    # Select features - Convert to array NumPy 2D
    y_quaternions = np.array(labels['quaternions'].to_list(), dtype=np.float32)
    y_transl_vect = np.array(labels['translation_vector'].to_list(), dtype=np.float32)
    y_outliers = np.array(labels['outliers'].to_list(), dtype=np.float32).reshape(-1, 1)

    # Load features to Dataset object
    ds_labels = tf.data.Dataset.from_tensor_slices({
        'quaternions': y_quaternions,
        'translations': y_transl_vect,
        'outliers': y_outliers
    })
    return ds_labels


# Data augmentation
def data_augmentation(image: tf.Tensor, labels: tf.data.Dataset) -> (tf.Tensor, tf.data.Dataset):
    """
    Applies data augmentation to the input images.

    :param image: (tf.Tensor): ['height', 'width', 'channels']
    :param labels: (tf.data.Dataset): (y_quaternions, y_transl_vect, y_outliers)
    :return: tf.Tensor: ['height', 'width', 'channels']
    :return: tf.data.Dataset: (y_quaternions, y_transl_vect, y_outliers)
    """

    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.02)
    return image, labels