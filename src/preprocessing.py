import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


# Convert from string to matrices and vectors
def adapt_format(string: str) -> list[float]:
    """
    Converts a string of 9 (representing a 3x3 matrix) or 3 (representing a vector)
    semicolon-separated values, into a vector represented by a list.

    Parameters:
        string (str): 'r11;r12;...;r33' or 'tx;ty;tz'

    Return:
        list[float]: [r11, r12, ..., r33] or [tx, ty, ..., tz]
    """
    elements = string.split(';')  # Split string
    elements_float = [float(element) for element in elements]  # Convert to float
    if len(elements) == 9:
        rotation_matrix = np.array(elements_float).reshape(3, 3)  # Convert to matrix
        return rotation_matrix
    else:
        return elements_float


# Define centre of camera calculation function
def compute_centre_camera(row: pd.Series) -> np.array:
    """
    From a row of the DataFrme with rotation matrix and translation vector,
    computes the centre of camera in each sample.
    """
    rot_mat = row['rotation_matrix']
    trans_vec = row['translation_vector']

    if np.isnan(rot_mat).any():
        return [np.nan, np.nan, np.nan]
    else:
        return -np.dot(rot_mat.T, trans_vec)


# Define rotation matrix to quaternion function
def rot2quat(rotation_matrix: list[float]) -> list[float]:
    """
    Converts a string of 9 semicolon-separated values, representing a 3x3 matrix, into a quaternion.

    Parameters:
        rotation_matrix (str): 'r11;r12;...;r33'

    Return:
        list[float]: [qx, qy, qz, qw]
    """

    if np.isnan(rotation_matrix).any():
        quaternions = [np.nan, np.nan, np.nan, np.nan]
    else:
        rotation_matrix = R.from_matrix(rotation_matrix)  # Extract R
        quaternions = rotation_matrix.as_quat()  # Convert to quaternions
    return quaternions


# Define quaternion to rotation matrix function
def quat2rot(quaternion_list: list[float]) -> str:
    """
    Converts a 4 elements list, representing a quaternion,
    into a string of 9 semicolon-separated values, representing a 3x3 matrix.

    Parameters:
        quaternion_list (list[float]): [qx, qy, qz, qw]

    Return:
        str: 'r11;r12;...;r33'
    """

    rot = R.from_quat(quaternion_list).as_matrix()  # Convert to matrix
    rot_str = ''.join([str(rot[row][col]) + ';' for row in range(rot.shape[0]) for col in range(rot.shape[1])])
    return rot_str[:-1]  # remove last character ';'


# Statistics extraction from labels
def statistics_from_data(input_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Extraction of statistical information (max and 95-percentile) from labels file
        - dist_tran = ||T_i - T_j|| (meters)
        - dist_rot = arccos(|dot(q_i, q_j)|) (rads)

    Parameters:
        input_dataframe (pd.Dataframe): columns('scene', 'quaternions', 'translation_vector')

    Return:
        pd.Dataframe: columns('scene', 'rot_max', 'rot_pct95', 'trans_max', 'trans_pct95')
    """

    dists = []
    for scene, grupo in input_dataframe.groupby('scene'):
        quats = np.stack(grupo['quaternions'].values)  # (M,4)
        tvecs = np.stack(grupo['translation_vector'].values)  # (M,3)
        M = quats.shape[0]

        for i in range(M):
            for j in range(i + 1, M):
                dot = abs(np.dot(quats[i], quats[j]))
                angle = np.arccos(np.clip(dot, -1.0, 1.0))  # radians
                d_trans = np.linalg.norm(tvecs[i] - tvecs[j])  # meters
                dists.append((scene, angle, d_trans))

    df_dists = pd.DataFrame(dists, columns=['scene', 'angle', 'd_trans'])

    # Per scene, max and 95-percentiles of angles and translation
    statistics = []
    for scene, g in df_dists.groupby('scene'):
        rot_max = g['angle'].max()  # Max angle observed in a scene
        rot_pct95 = np.percentile(g['angle'], 95)
        trans_max = g['d_trans'].max()  # Max distance observed in a scene
        trans_pct95 = np.percentile(g['d_trans'], 95)
        statistics.append((scene, rot_max, rot_pct95, trans_max, trans_pct95))
    df_statistics = pd.DataFrame(statistics, columns=['scene', 'rot_max', 'rot_pct95', 'trans_max', 'trans_pct95'])
    return df_statistics


def normalize_transl_vector(df: pd.DataFrame, scaler):
    """
    Normalization of only valid (not NaN) 'translation_vector' samples

    :param df: DataFrame with columns (..., 'outliers', 'translation_vector', ...)
    :param scaler: StandardScaler() from Scikit-learn for translation vector
    :return:
    """

    inlier_mask = (df['outliers'] == 0)
    transl_valid_samples_scaled = scaler.transform(
        np.array(df.loc[inlier_mask, 'translation_vector'].to_list(), dtype=np.float32)
    ).tolist()
    df.loc[inlier_mask, 'translation_vector'] = pd.Series(transl_valid_samples_scaled, index=df[inlier_mask].index)