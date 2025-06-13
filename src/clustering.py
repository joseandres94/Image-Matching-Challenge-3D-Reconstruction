import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN


# Clusters prediction function
def clusters_prediction(predictions, visual_features, threshold_rot, threshold_trans):
    """
        Computes DBSCAN with pose predictions and features extracted from images.
        Clusters are predicted based on rotational and translational thresholds and a distance between samples (epsilon)
    """

    # Inlier masks
    inlier_mask = (predictions['outliers'].round().squeeze() == 0)
    outlier_mask = ~inlier_mask

    # Normalize positional features (only for inliers)
    quat_inliers = predictions['quaternions'][inlier_mask]
    trans_inliers = predictions['translations'][inlier_mask]
    quat_inliers_norm = quat_inliers / threshold_rot
    trans_inliers_norm = trans_inliers / threshold_trans

    # Visual features (only for inliers)
    scaler_vis = StandardScaler()
    V_norm = scaler_vis.fit_transform(visual_features[inlier_mask])

    # Normalization of all features
    X_feats = np.concatenate([quat_inliers_norm, trans_inliers_norm, V_norm], axis=1)
    scaler_dbscan = StandardScaler()
    X_feats_scaled = scaler_dbscan.fit_transform(X_feats)

    # DBSCAN algorithm
    distances = pairwise_distances(X_feats_scaled)
    avg_dist = np.mean(np.sort(distances, axis=1)[:, 1])  # First real neighbour
    clusters = DBSCAN(eps=avg_dist * 1.5, min_samples=3, metric='euclidean').fit(X_feats_scaled)
    labels = clusters.labels_

    # Join predicted outliers from model
    for i in range(len(outlier_mask)):
        if outlier_mask[i]:
            labels[i] = -1
    return labels