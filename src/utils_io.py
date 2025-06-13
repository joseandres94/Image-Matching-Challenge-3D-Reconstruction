from src.preprocessing import quat2rot

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf


def load_train_files(paths) -> (pd.DataFrame,  pd.DataFrame):
    """
    Provided the input path, this function reads the labels and thresholds files

    :param paths: Path of input folder
    :return: train_labels_file (pd.DataFrame): DataFrame with labels provided
    :return: train_thresholds_file (pd.DataFrame): DataFrame with thresholds values provided
    """

    train_labels_file = pd.read_csv(paths.TRAIN_LABELS_FILE)
    train_thresholds_file = pd.read_csv(paths.TRAIN_THRESHOLDS_FILE)

    return train_labels_file, train_thresholds_file


def load_and_preprocess_images(image_path: tf.Tensor, size_image: tuple[int, int]) -> tf.Tensor:
    """
    Provided the path of a image file, decodes the image in RGB and resizes it to 'size_image'.

    :param image_path: Path for each image
    :param size_image: ('new_height', 'new_width')
    :return: image: ['new_height', 'new_width', 'channels']
    """

    raw_image = tf.io.read_file(image_path)  # check type of image_path (if string or tf.Tensor)
    image = tf.image.decode_image(raw_image, channels=3)
    # Resize image
    image = tf.ensure_shape(image, (None, None, 3))
    image = tf.image.resize(image, size_image)
    return image


def create_submission_file(folder_path: Path,
                           list_images_path: list[Path],
                           labels: np.array,
                           predictions: dict,
                           path) -> pd.DataFrame:
    """
    Creates a pd.DataFrame with the predictions in the required format

    :param folder_path: Output folder path
    :param list_images_path: List of image paths
    :param labels: Predicted clusters
    :param predictions: Dictionary with predictions
    :param path: Output folder path
    :return: sample_submission (pd.DataFrame): DataFrame with the predictions
    """

    # Image_id and Dataset columns
    func = lambda x: (str(x) + '_public').replace(str(folder_path) + '/', '').split('/')
    image_id = pd.DataFrame(list(map(func, list_images_path)), columns = ['dataset', 'image_id'])
    image_id = image_id[['image_id', 'dataset']]

    # Scene predictions column (string)
    func = lambda x: 'cluster' + str(x)
    scene = pd.Series(list(map(func, labels)), name = 'scene').apply(lambda x: x.replace('cluster-1', 'outliers'))

    # Image column
    func = lambda x: str(x).replace(str(folder_path), '').split('/')[2]
    image = pd.Series(list(map(func, list_images_path)), name = 'image')

    # Rotation matrix to string column
    rot_matrix_pred = pd.Series(list(map(quat2rot, predictions['quaternions'])), name = 'rotation_matrix')

    # Translation vector to string column
    func = lambda x: ''.join([str(x[i]) + ';' for i in range(len(x))])[:-1]
    transl_vec_pred = pd.Series(list(map(func, predictions['translations'])), name = 'translation_vector')

    # Sample submission dataframe
    sample_submission = pd.concat([image_id, scene, image, rot_matrix_pred, transl_vec_pred], axis = 1)

    # Set nan for Rotation matrix and Translation vector of outliers predictions
    outliers = predictions['outliers'].round().squeeze() == 1
    sample_submission['rotation_matrix'] = sample_submission['rotation_matrix'].\
        mask(pd.Series(outliers), ['nan;nan;nan;nan;nan;nan;nan;nan;nan'])
    sample_submission['translation_vector'] = sample_submission['translation_vector'].\
        mask(pd.Series(outliers), ['nan;nan;nan'])

    # Convert to CSV
    sample_submission.to_csv(path.OUTPUT_PATH / 'submission.csv', index=False)
    return sample_submission
