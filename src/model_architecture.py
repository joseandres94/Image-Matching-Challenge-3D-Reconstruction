from src import config

import tensorflow as tf
from tensorflow.keras import layers, regularizers
from src.losses import QuaternionAngularLoss, TranslationLoss, CameraCenterLoss
from src.metrics import QuaternionCosineSimilarityMetric, TranslMAEMetric
from tensorflow.keras.applications import ConvNeXtLarge


# - Classes definition - #
# Custom layers
class L2NormalizationLayer(layers.Layer):
    """
    This layer applies L2 normalization to the predicted quaternions.
    """

    def __init__(self, **kwargs):
        super(L2NormalizationLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=-1)  # Normalizes along the quaternions dimension

    def get_config(self):
        # Call get_config to make it serializable
        config = super(L2NormalizationLayer, self).get_config()
        return config


# Custom subclass model
class Model(tf.keras.Model):
    def __init__(self, base_model, quat_head, trans_head, outlier_head, trans_mean, trans_scale, batch_size):
        super().__init__()
        # Instantiation of heads
        self.backbone = base_model
        self.head_q = quat_head
        self.head_t = trans_head
        self.head_o = outlier_head

        # Instantiation of loss functions
        self.quat_loss_fn = QuaternionAngularLoss(global_batch_size=batch_size)
        self.trans_loss_fn = TranslationLoss(global_batch_size=batch_size,
                                             trans_mean=trans_mean,
                                             trans_scale=trans_scale)
        self.outlier_loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.center_loss_fn = CameraCenterLoss(global_batch_size=batch_size,
                                               trans_mean=trans_mean,
                                               trans_scale=trans_scale)

        # Instantiation of metrics
        self.quat_metric = QuaternionCosineSimilarityMetric(name='quaternions_cos_similarity')
        self.trans_metric = TranslMAEMetric(name='translations_mae')
        self.outlier_accuracy = tf.keras.metrics.BinaryAccuracy(name='outliers_binary_accuracy')

    def call(self, x):
        # Definition of forward pass
        features = self.backbone(x)
        quaternions = self.head_q(features)
        translations = self.head_t(features)
        outliers = self.head_o(features)
        return {
            'quaternions': quaternions,
            'translations': translations,
            'outliers': outliers
        }

    def train_step(self, data):
        # Unpack input data
        x, y_true = data
        with tf.GradientTape() as tape:
            # Get predictions (forward pass)
            y_pred = self(x, training=True)

            # Get individual losses
            loss_quat = self.quat_loss_fn(y_true['quaternions'], y_pred['quaternions'])
            loss_trans = self.trans_loss_fn(y_true['translations'], y_pred['translations'])
            loss_outlier = self.outlier_loss_fn(y_true['outliers'], y_pred['outliers'])
            loss_centres = self.center_loss_fn(y_true, y_pred)

            # Get global loss
            global_loss = loss_quat * config.W_QUAT + \
                          loss_trans * config.W_TRANS + \
                          loss_outlier * config.W_OUTL + \
                          loss_centres * config.W_CENT

        grads = tape.gradient(global_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update metrics
        self.quat_metric.update_state(y_true['quaternions'], y_pred['quaternions'])
        self.trans_metric.update_state(y_true['translations'], y_pred['translations'])
        self.outlier_accuracy.update_state(y_true['outliers'], y_pred['outliers'])

        return {'loss': global_loss,
                'quaternions_loss': loss_quat,
                'translations_loss': loss_trans,
                'outliers_loss': loss_outlier,
                'centres_loss': loss_centres,
                self.quat_metric.name: self.quat_metric.result(),
                self.trans_metric.name: self.trans_metric.result(),
                self.outlier_accuracy.name: self.outlier_accuracy.result()}

    def test_step(self, data):
        x, y_true = data

        # Get predictions (forward pass)
        y_pred = self(x, training=False)

        # Get individual losses
        loss_quat = self.quat_loss_fn(y_true['quaternions'], y_pred['quaternions'])
        loss_trans = self.trans_loss_fn(y_true['translations'], y_pred['translations'])
        loss_outlier = self.outlier_loss_fn(y_true['outliers'], y_pred['outliers'])
        loss_centres = self.center_loss_fn(y_true, y_pred)

        # Get global loss
        global_loss = loss_quat * config.W_QUAT + \
                      loss_trans * config.W_TRANS + \
                      loss_outlier * config.W_OUTL + \
                      loss_centres * config.W_CENT

        # Update metrics
        self.quat_metric.update_state(y_true['quaternions'], y_pred['quaternions'])
        self.trans_metric.update_state(y_true['translations'], y_pred['translations'])
        self.outlier_accuracy.update_state(y_true['outliers'], y_pred['outliers'])

        return {
            'loss': global_loss,
            'quaternions_loss': loss_quat,
            'translations_loss': loss_trans,
            'outliers_loss': loss_outlier,
            'centres_loss': loss_centres,
            self.quat_metric.name: self.quat_metric.result(),
            self.trans_metric.name: self.trans_metric.result(),
            self.outlier_accuracy.name: self.outlier_accuracy.result()
        }

    @property
    def metrics(self):
        return [self.quat_metric, self.trans_metric, self.outlier_accuracy]


# - Functions definition - #
# Backbone definition
def build_backbone() -> tf.keras.Model:
    """
    Model 'ConvNeXtLarge' loading.

    Parameters:

    Return:
        tf.keras.Model
    """

    # Import base model - ConvNeXt
    base_model = ConvNeXtLarge(
        include_top=False,
        include_preprocessing=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    # Set all layers not trainable, for first stage of training
    base_model.trainable = False

    # Build base model
    backbone = tf.keras.Model(inputs=base_model.input, outputs=base_model.output, name='backbone')
    return backbone


# Heads - Quaternions regressor
def create_quat_head(input_shape: int, dropout_rate: float, l2_reg: float):
    """
    Attaches a head to a backbone model for quaternions predictions.
    The head is made of Dropout, Dense and BatchNormalization layers.

    Parameters:
        input_shape (int): Input shape of feature vectors from backbone model.
        dropout_rate (float): Percentage of connections removed during training.
        l2_reg (float): L2 regularization penalty for output layer.

    Return:
        tf.keras.Sequential
    """
    return tf.keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dropout(dropout_rate),
        layers.BatchNormalization(),
        layers.Dense(units=4, name='quaternions_raw',
                     kernel_regularizer=regularizers.l2(l2_reg)),
        L2NormalizationLayer(name='quaternions')
    ], name='quaternions_head')


# Heads -  Translation vectors regressor
def create_trans_head(input_shape: int, dropout_rate: float, l2_reg: float):
    """
    Attaches a head to a backbone model for translation vector predictions.
    The head is made of Dropout, Dense and BatchNormalization layers.

    Parameters:
        input_shape (int): Input shape of feature vectors from backbone model.
        dropout_rate (float): Percentage of connections removed during training.
        l2_reg (float): L2 regularization penalty for output layer.

    Return:
        tf.keras.Sequential
    """
    return tf.keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dropout(dropout_rate),
        layers.BatchNormalization(),
        layers.Dense(units=3, name='translations',
                     kernel_regularizer=regularizers.l2(l2_reg))
    ], name='translations_head')


# Heads -  Outliers classification
def create_outl_head(input_shape: int, dropout_rate: float, l2_reg: float):
    """
    Attaches a head to a backbone model for outlier predictions.
    The head is made of Dropout, Dense and BatchNormalization layers.

    Parameters:
        input_shape (int): Input shape of feature vectors from backbone model.
        dropout_rate (float): Percentage of connections removed during training.
        l2_reg (float): L2 regularization penalty for output layer.

    Return:
        tf.keras.Sequential
    """
    return tf.keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dropout(dropout_rate),
        layers.BatchNormalization(),
        layers.Dense(units=1, activation='sigmoid', name='outliers',
                     kernel_regularizer=regularizers.l2(l2_reg))
    ], name='outliers_head')
