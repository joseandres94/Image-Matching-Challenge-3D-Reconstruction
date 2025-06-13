import numpy as np

# Joblib
from joblib import load


def run_inference(feature_extractor, model, test_dataset, path):
    """
    Extract pose predictions and visual features  from a test dataset

    :param feature_extractor:
    :param model:
    :param test_dataset:
    :param path:
    :return:
    """

    visual_features = feature_extractor.predict(test_dataset)
    predictions = model.predict(test_dataset)

    # Revert normalization of translation predictions
    # Load trans_scaler (to prevent possible overwrittings)
    scaler = load(path.SCALER_PATH)
    predictions['translations'] = scaler.inverse_transform(predictions['translations'])
    # Normalization of quaternions
    norms = np.linalg.norm(predictions['quaternions'], axis=1, keepdims=True)
    predictions['quaternions'] = predictions['quaternions'] / np.maximum(norms, 1e-6)

    return visual_features, predictions
