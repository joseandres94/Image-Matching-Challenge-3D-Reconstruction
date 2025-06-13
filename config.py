# -------------------------------------------------
# 1. VARIABLES DEFINITION
# -------------------------------------------------

# Libraries
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import mixed_precision

# Paths
INPUT_PATH = Path('/kaggle/input/image-matching-challenge-2025')
OUTPUT_PATH = Path('/kaggle/working')
TRAIN_LABELS_FILE = INPUT_PATH / 'train_labels.csv'
TRAIN_THRESHOLDS_FILE = INPUT_PATH / 'train_thresholds.csv'
TRAIN_FOLDER = INPUT_PATH / 'train'
TEST_FOLDER = INPUT_PATH / 'test'
SCALER_PATH = OUTPUT_PATH / 'translation_vector_scaler.bin'

# Hyperparameters
SEED = 40  # Seeds during shuffling
BATCH_SIZE_MULTIPLIER = 16
IMAGE_SIZE = (224, 224)
# Model Architecture
N_UNITS_HEAD = 256
DROPOUT_RATE = 0.3
L2_REG = 1e-3
# Training
ES_PATIENCE = 5
NUM_EPOCHS = 40
W_QUAT = 10.0
W_TRANS = 0.01
W_OUTL = 10.0
W_CENT = 0.001
# Training - Phase 1 (easy images)
WARMUP_STEPS_RATE = 0.2
LR_INITIAL_WARMUP = 1e-08
LR_TARGET_WARMUP = 1e-04
# Training - Phase 2 (all images)
LR_INITIAL = 1e-05
# Fine-tuning - Phase 1
FROZEN_LAYERS_RATE_1 = 0.8
NUM_EPOCHS_FT1 = 20  # 5
LR_INITIAL_FT1 = 1e-06
# Fine-tuning - Phase 2
FROZEN_LAYERS_RATE_2 = 0.6
NUM_EPOCHS_FT2 = 20  # 5
LR_INITIAL_FT2 = 1e-07


def configure_strategy_and_batch_size(mixed_float: bool) -> (tf.distribute.TPUStrategy, int):
    """
    Configuration of strategy for training and its corresponding batch size

    :param mixed_float: Boolean to activate 'mixed_float' policy
    :return: strategy: Strategy available for training
    :return: batch_size: Size of batch
    """

    if mixed_float:
        # Set mixed precision
        mixed_precision.set_global_policy('mixed_float16')

    try:
        # Detection and connection to TPU
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("TPU connected")
        batch_size = BATCH_SIZE_MULTIPLIER * strategy.num_replicas_in_sync  # Set batch size
        return strategy, batch_size

    except (ValueError, tf.errors.NotFoundError, tf.errors.InvalidArgumentError):
        # In case of no TPU available, use CPU/GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                strategy = tf.distribute.MirroredStrategy()  # Distributed strategy in case of more GPUs available
                print("GPU connected. Using: ", strategy)
                print("Number of replicas in sync: ", strategy.num_replicas_in_sync)
                batch_size = BATCH_SIZE_MULTIPLIER * strategy.num_replicas_in_sync  # Set batch size
                return strategy, batch_size

            except RuntimeError as e:
                # Errores de inicialización de GPU
                print(e)
                strategy = tf.distribute.get_strategy()  # Using default strategy for CPU
                print("Error setting GPU, using: ", strategy)
                batch_size = BATCH_SIZE_MULTIPLIER  # Set batch size
                return strategy, batch_size
        else:
            # Si no hay GPU, usar CPU
            strategy = tf.distribute.get_strategy()
            print("TPU nor GPU are available, using: ", strategy)
            batch_size = BATCH_SIZE_MULTIPLIER  # Set batch size
            return strategy, batch_size
