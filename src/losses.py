import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg


# - Classes definition - #
# Loss function for quaternions output
class QuaternionAngularLoss(tf.keras.losses.Loss):
    """
    This custom loss function computes the loss based on the similarity
    of the predicted and real quaternion, by:
        Loss = 1 - |y_true * y_pred| (per sample)
        Average Loss = sum(Loss_sample * Weight_sample)/(Global Batch Size * Num_Replicas)
    """

    def __init__(self, global_batch_size, name="angular_loss"):
        # Definition of reduction
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name)
        self._gbs = global_batch_size

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Filter NaNs from inputs
        is_valid = tf.reduce_all(~tf.math.is_nan(y_true), axis=-1)  # Check if valid
        y_true = tf.where(tf.expand_dims(is_valid, axis=-1), y_true, tf.zeros_like(y_true))

        # Calculate angular similarity loss per sample
        # Equation: Error = 1 - |y_true * y_pred|
        y_true = tf.math.l2_normalize(y_true, axis=-1)
        y_pred = tf.math.l2_normalize(y_pred, axis=-1)
        loss_per_sample = 1 - tf.abs(tf.reduce_sum(y_true * y_pred, axis=-1))

        # Weight per element
        weight_per_sample = tf.cast(is_valid, tf.float32)
        # Scale weights (global_batch_size / valid_samples)
        n_valid_samples = tf.maximum(tf.reduce_sum(weight_per_sample), 1.0)
        scale = tf.cast(self._gbs, tf.float32) / n_valid_samples
        weight_scaled = weight_per_sample * scale

        # Compute the average loss for only the valid samples (using weight_scaled)
        local_average = tf.nn.compute_average_loss(
            loss_per_sample,
            sample_weight=weight_scaled,
            global_batch_size=self._gbs
        )

        # Keras sums the local average for all replicas,
        # so it is necessary to divide by num_replicas
        num_replicas = tf.cast(tf.distribute.get_strategy().num_replicas_in_sync,
                               tf.float32)
        return local_average / num_replicas

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "global_batch_size": self._gbs}


# Loss function for translation vectors output
class TranslationLoss(tf.keras.losses.Loss):
    """
    This custom loss function computes the loss based on the Mean Squared Error
    of the predicted and real translation vectors, by:
        Loss = (y_true - y_pred)^2 (per sample)
        Average Loss = sum(Loss_sample * Weight_sample)/(Global Batch Size * Num_Replicas)
    """

    def __init__(self, global_batch_size, trans_mean, trans_scale, name='mse_loss'):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name)
        self._gbs = global_batch_size
        self.mean = tf.constant(trans_mean, dtype=tf.float32)
        self.scale = tf.constant(trans_scale, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Filter NaNs from inputs
        is_valid = tf.reduce_all(~tf.math.is_nan(y_true), axis=-1)  # Check if valid
        y_true = tf.where(tf.expand_dims(is_valid, axis=-1), y_true, tf.zeros_like(y_true))

        # Reverse translation vectors normalization to real scale
        # Equation: transl = (transl_norm * scale) + mean
        y_true = (y_true * self.scale) + self.mean
        y_pred = (y_pred * self.scale) + self.mean

        # Calculate Translation MSE (Sum of Squares) per sample
        # Equation: Error = (y_true - y_pred)^2
        loss_per_sample = tf.square(y_true - y_pred)  # Square element-wise
        loss_per_sample = tf.reduce_mean(loss_per_sample, axis=-1)

        # Weight per element
        weight_per_sample = tf.cast(is_valid, tf.float32)
        # Scale weights (global_batch_size / valid_samples)
        n_valid_samples = tf.maximum(tf.reduce_sum(tf.cast(is_valid, tf.float32)), 1.0)
        scale = tf.cast(self._gbs, tf.float32) / n_valid_samples
        weight_scaled = weight_per_sample * scale

        # Compute the average loss for only the valid samples (using weight_scaled)
        local_average = tf.nn.compute_average_loss(
            loss_per_sample,
            sample_weight=weight_scaled,
            global_batch_size=self._gbs
        )

        # Keras sums the local average for all replicas,
        # so it is necessary to divide by num_replicas
        num_replicas = tf.cast(tf.distribute.get_strategy().num_replicas_in_sync,
                               tf.float32)
        return local_average / num_replicas

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "global_batch_size": self._gbs}


# Loss function for camera centre vectors
class CameraCenterLoss(tf.keras.losses.Loss):
    """
    This custom loss function computes the loss based on the Mean Squared Error
    of the predicted and real centres of camera of each scene, by:
        Loss = 1 - (C_true - C_pred)^2 (per sample)
        Average Loss = sum(Loss_sample * Weight_sample)/(Global Batch Size * Num_Replicas)
    """

    def __init__(self, global_batch_size, trans_mean, trans_scale, name='centre_loss'):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name)
        self._gbs = global_batch_size
        self.mean = tf.constant(trans_mean, dtype=tf.float32)
        self.scale = tf.constant(trans_scale, dtype=tf.float32)

    def call(self, y_true, y_pred):
        # Unpack data
        y_quat_true = y_true['quaternions']
        y_quat_pred = y_pred['quaternions']
        y_trans_true = y_true['translations']
        y_trans_pred = y_pred['translations']

        # Cast variables to float32
        y_quat_true = tf.cast(y_quat_true, tf.float32)
        y_quat_pred = tf.cast(y_quat_pred, tf.float32)
        y_trans_true = tf.cast(y_trans_true, tf.float32)
        y_trans_pred = tf.cast(y_trans_pred, tf.float32)

        # Filter NaNs from inputs
        is_valid = tf.reduce_all(~tf.math.is_nan(y_quat_true), axis=-1)  # Check if valid
        y_quat_true = tf.where(tf.expand_dims(is_valid, axis=-1), y_quat_true, tf.zeros_like(y_quat_true))
        y_trans_true = tf.where(tf.expand_dims(is_valid, axis=-1), y_trans_true, tf.zeros_like(y_trans_true))

        # Convert quaternions to rotation matrices
        y_quat_true = tf.math.l2_normalize(y_quat_true, axis=-1)
        y_quat_pred = tf.math.l2_normalize(y_quat_pred, axis=-1)
        y_quat_true = tfg.rotation_matrix_3d.from_quaternion(y_quat_true)
        y_quat_pred = tfg.rotation_matrix_3d.from_quaternion(y_quat_pred)

        # Reverse translation vectors normalization to real scale
        # Equation: transl = (transl_norm * scale) + mean
        y_trans_true = (y_trans_true * self.scale) + self.mean
        y_trans_pred = (y_trans_pred * self.scale) + self.mean

        # Expand dimensions for translation vectors
        y_trans_true = tf.expand_dims(y_trans_true, axis=-1)
        y_trans_pred = tf.expand_dims(y_trans_pred, axis=-1)

        # Calculate loss per sample
        # Equation: Centre_camera = -transpose(R)*T
        C_true = -tf.linalg.matmul(y_quat_true, y_trans_true, transpose_a=True)
        C_pred = -tf.linalg.matmul(y_quat_pred, y_trans_pred, transpose_a=True)
        loss_per_sample = tf.square(C_true - C_pred)  # Square element-wise
        loss_per_sample = tf.reduce_mean(loss_per_sample, axis=-1)

        # Weight per element
        weight_per_sample = tf.cast(is_valid, tf.float32)
        # Scale weights (global_batch_size / valid_samples)
        n_valid_samples = tf.maximum(tf.reduce_sum(tf.cast(is_valid, tf.float32)), 1.0)
        scale = tf.cast(self._gbs, tf.float32) / n_valid_samples
        weight_scaled = weight_per_sample * scale

        # Compute the average loss for only the valid samples (using weight_scaled)
        local_average = tf.nn.compute_average_loss(
            loss_per_sample,
            sample_weight=weight_scaled,
            global_batch_size=self._gbs
        )

        # Keras sums the local average for all replicas,
        # so it is necessary to divide by num_replicas
        num_replicas = tf.cast(tf.distribute.get_strategy().num_replicas_in_sync,
                               tf.float32)
        return local_average / num_replicas

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,  # Serialization to save model
                "global_batch_size": self._gbs,
                "translation_mean": self.mean.numpy().tolist(),
                "translation_scale": self.scale.numpy().tolist()
                }
