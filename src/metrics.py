import tensorflow as tf


# - Classes definition - #
# Metric for quaternions output
class QuaternionCosineSimilarityMetric(tf.keras.metrics.Metric):
    """
    This custom metric computes the similarity of the predicted and real quaternion, by:
        Similarity = |y_true * y_pred| (per sample)
        Average Similarity = sum(Similarity)/Num_Valid_Samples
    """

    def __init__(self, name='cos_similarity', dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.total_similarity = self.add_weight(name='total_similarity', initializer='zeros', dtype=dtype)
        self.count_valid = self.add_weight(name='count', initializer='zeros', dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Filter NaNs from inputs
        is_valid = tf.reduce_all(~tf.math.is_nan(y_true), axis=-1)  # Check if valid
        y_true = tf.boolean_mask(y_true, is_valid)
        y_pred = tf.boolean_mask(y_pred, is_valid)

        # Calculate cosine similarity per sample (equal as the dot product for
        # normalized vectors)
        # Equation: Similarity = |y_true * y_pred|
        y_true = tf.math.l2_normalize(y_true, axis=-1)
        y_pred = tf.math.l2_normalize(y_pred, axis=-1)
        similarity_per_sample = tf.abs(tf.reduce_sum(y_true * y_pred, axis=-1))

        # Addition of all samples in batch
        self.total_similarity.assign_add(tf.reduce_sum(similarity_per_sample))
        # Count of valid samples in batch
        count = tf.cast(tf.shape(y_true)[0], self.dtype)
        self.count_valid.assign_add(tf.cast(tf.maximum(count, 1), self.dtype))

    def result(self):
        # Average error
        return tf.divide(self.total_similarity, self.count_valid)

    def reset_state(self):
        # Reset values
        self.total_similarity.assign(0.0)
        self.count_valid.assign(0.0)

    def get_config(self):
        # Assure serialization
        base_config = super().get_config()
        return {**base_config}


# Metric for translation vectors output
class TranslMAEMetric(tf.keras.metrics.Metric):
    """
    This custom metric computes the Mean Absolute Error of the predicted and real translation vectors, by:
        MAE = |y_true - y_pred| (per sample)
        Average Similarity = sum(MAE)/Num_Valid_Samples
    """
    def __init__(self, name='mae', dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.total_abs_error = self.add_weight(name='total_abs_error', initializer='zeros', dtype=dtype)
        self.count_valid = self.add_weight(name='count', initializer='zeros', dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Filter NaNs from inputs
        is_valid = tf.reduce_all(~tf.math.is_nan(y_true), axis=-1)  # Check if valid
        y_true = tf.boolean_mask(y_true, is_valid)
        y_pred = tf.boolean_mask(y_pred, is_valid)

        # Calculate Translation MAE (Mean Absolute Error) per sample
        # Equation: Error = |y_true - y_pred|
        error_per_sample = tf.reduce_mean(tf.abs(y_true - y_pred), axis=-1)

        # Addition of all values in batch
        self.total_abs_error.assign_add(tf.reduce_sum(error_per_sample))
        # Count of valid samples in batch
        count = tf.cast(tf.shape(y_true)[0], self.dtype)
        self.count_valid.assign_add(tf.cast(tf.maximum(count, 1), self.dtype))

    def result(self):
        # Average error
        return tf.divide(self.total_abs_error, self.count_valid)

    def reset_state(self):
        # Reset values
        self.total_abs_error.assign(0.0)
        self.count_valid.assign(0.0)

    def get_config(self):
        # Assure serialization
        base_config = super().get_config()
        return {**base_config}


# - Functions definition - #
# Manual metric computation (with TPU, online metrics produces errors)
def compute_metrics(valid_data: tf.data.Dataset, model: tf.keras.Model):
    """
    When using TPU, compute the metrics after each phase of training.

    """

    quat_metric = QuaternionCosineSimilarityMetric()
    transl_metric = TranslMAEMetric()
    outliers_metric = tf.keras.metrics.BinaryAccuracy()

    for x, y_true in valid_data:
        y_pred = model.predict(x, verbose=0)

        # Quaternion Cosine Similarity
        quat_metric.update_state(
            y_true=y_true['quaternions'],
            y_pred=y_pred['quaternions']
        )

        # Translation MAE Metric
        transl_metric.update_state(
            y_true=y_true['translations'],
            y_pred=y_pred['translations']
        )

        # Outliers Binary Accuracy
        outliers_metric.update_state(
            y_true=y_true['outliers'],
            y_pred=y_pred['outliers']
        )

    # Print result of metrics
    quat_sim_val = quat_metric.result().numpy()
    transl_mae_val = transl_metric.result().numpy()
    outliers_acc_val = outliers_metric.result().numpy()

    print(f"- Quaternions Cosine Similarity: {quat_sim_val:.4f}")
    print(f"- Translation MAE: {transl_mae_val:.4f}")
    print(f"- Binary precision of Outliers: {outliers_acc_val:.4f}")

    # Reset state of metric objects
    quat_metric.reset_state()
    transl_metric.reset_state()
    outliers_metric.reset_state()