from src import config
from src.utils_io import load_train_files, load_and_preprocess_images, create_submission_file
from src.preprocessing import adapt_format, compute_centre_camera, rot2quat, statistics_from_data,\
    normalize_transl_vector
from src.dataset import create_labels_dataset, data_augmentation
from src.model_architecture import Model, build_backbone, create_quat_head, create_trans_head, create_outl_head
from src.training import early_stopping_callback, learning_rate_cosine_decay, training_model
from src.plot_results import plot_results, plot_clusters
from src.inference import run_inference
from src.clustering import clusters_prediction


# -------------------------------------------------
# 0. IMPORT LIBRARIES
# -------------------------------------------------
# Standard Python libraries
import math

# Data manipulation
import pandas as pd
import numpy as np

# TensorFlow and Keras
import tensorflow as tf

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Joblib
from joblib import dump


def main():
    # -------------------------------------------------
    # 1. CONFIGURATION OF TF FOR USING TPU/GPU/CPU
    # -------------------------------------------------
    print('---- Defining CPU/GPU/TPU strategy and batch size ----')
    # Strategy and batch size configuration
    strategy, batch_size=config.configure_strategy_and_batch_size(mixed_float=True)
    print(f"Final batch size: {batch_size}")


    # -------------------------------------------------
    # 2. LOAD DATA
    # -------------------------------------------------
    print('---- Loading train labels ----')
    # Loading train data
    train_labels_file, train_thresholds_file=load_train_files(config)

    # -------------------------------------------------
    # 3. DATA PREPROCESSING
    # -------------------------------------------------
    print('---- Preprocessing train labels ----')
    # Create binary column based on outliers
    train_labels_file['outliers'] = train_labels_file['scene'].apply(lambda x: 1 if x == 'outliers' else 0)

    # Convert rotation matrices and translation vectors from string to list
    train_labels_file['rotation_matrix'] = train_labels_file['rotation_matrix'].apply(lambda x: adapt_format(x))
    train_labels_file['translation_vector'] = train_labels_file['translation_vector'].apply(lambda x: adapt_format(x))

    # Compute centres of camera of each image
    train_labels_file['camera_centre'] = train_labels_file.apply(lambda x: compute_centre_camera(x), axis=1)

    # Compute difficulty of scene based on thresholds file (from 1 to 100)
    train_thresholds_file['thresholds'] = train_thresholds_file['thresholds'].apply(lambda x: adapt_format(x))
    train_thresholds_file['difficulty'] = train_thresholds_file['thresholds'].apply(lambda x: 1 / (min(x) + 1e-2))

    # Move thresholds and difficulty to train_labels_file dataframe
    train_labels_file = pd.merge(train_labels_file,
                                 train_thresholds_file[['scene', 'thresholds', 'difficulty']],
                                 on='scene',
                                 how='left')

    # Append quaternions to train dataframe
    train_labels_file['quaternions'] = train_labels_file['rotation_matrix'].apply(lambda x: rot2quat(x))

    # Extraction of max and 95-percentiles values from quaternions and translation vectors
    stats=statistics_from_data(train_labels_file)

    # -------------------------------------------------
    # 4. SPLIT DATA (TRAIN/VALID)
    # -------------------------------------------------
    print('---- Splitting data into easy_train/train/valid datasets ----')
    # Split labels to train/valid data
    train_labels, valid_labels = train_test_split(train_labels_file,
                                                  train_size=0.8,
                                                  shuffle=True,
                                                  random_state=config.SEED,
                                                  stratify=train_labels_file['scene'])  # To keep scenes ratio

    # Split train_labels based on difficulty of scenes
    inlier_mask = (train_labels['outliers'] == 0)  # Extract only inlier samples
    average_difficulty = int(train_labels.loc[inlier_mask, 'difficulty'].mean())  # Average value of difficulty per dataset
    # Extraction of datasets with lowest difficulty (lower than average level)
    easy_datasets = train_labels.loc[inlier_mask].groupby('dataset')['difficulty'].mean() < average_difficulty
    easy_train_labels = train_labels[train_labels['dataset'].isin(easy_datasets[easy_datasets == True].index)].copy()

    # Normalization of translation vectors by using StandardScaler() from scikit-learn
    # Applied only for inliers in order to prevent scaler from contaminating with NaNs
    transl_scaler = StandardScaler()
    transl_scaler.fit(np.array(train_labels.loc[inlier_mask, 'translation_vector'].to_list(), dtype=np.float32))

    # Apply normalization for each set of labels
    normalize_transl_vector(easy_train_labels, transl_scaler)
    normalize_transl_vector(train_labels, transl_scaler)
    normalize_transl_vector(valid_labels, transl_scaler)

    # Save scaler from translation vector normalization
    dump(transl_scaler, config.SCALER_PATH, compress=True)


    # -------------------------------------------------
    # 5. LOAD IMAGES
    # -------------------------------------------------
    print('---- Loading images ----')
    # Obtain list of paths for TRAIN images
    list_easy_train_images_path = (config.TRAIN_FOLDER / easy_train_labels['dataset'] / easy_train_labels['image']).\
        to_list()
    list_train_images_path = (config.TRAIN_FOLDER / train_labels['dataset'] / train_labels['image']).to_list()
    list_valid_images_path = (config.TRAIN_FOLDER / valid_labels['dataset'] / valid_labels['image']).to_list()

    # Obtain list of paths for TEST images
    list_test_images_path = list(config.TEST_FOLDER.rglob('*.png'))

    # Create Dataset object from list_train_images (previously converted from PosixPath to strings)
    ds_easy_train_images_path = tf.data.Dataset.from_tensor_slices([str(path) for path in list_easy_train_images_path])
    ds_train_images_path = tf.data.Dataset.from_tensor_slices([str(path) for path in list_train_images_path])
    ds_valid_images_path = tf.data.Dataset.from_tensor_slices([str(path) for path in list_valid_images_path])
    ds_test_images_path = tf.data.Dataset.from_tensor_slices([str(path) for path in list_test_images_path])

    # Load images to Dataset object
    ds_easy_train_images = ds_easy_train_images_path.map(lambda x: load_and_preprocess_images(x, config.IMAGE_SIZE),
                                                         num_parallel_calls=tf.data.AUTOTUNE)
    ds_train_images = ds_train_images_path.map(lambda x: load_and_preprocess_images(x, config.IMAGE_SIZE),
                                               num_parallel_calls=tf.data.AUTOTUNE)
    ds_valid_images = ds_valid_images_path.map(lambda x: load_and_preprocess_images(x, config.IMAGE_SIZE),
                                               num_parallel_calls=tf.data.AUTOTUNE)
    ds_test_images = ds_test_images_path.map(lambda x: load_and_preprocess_images(x, config.IMAGE_SIZE),
                                             num_parallel_calls=tf.data.AUTOTUNE)


    # -------------------------------------------------
    # 6. DATASETS CREATION
    # -------------------------------------------------
    print('---- Datasets construction ----')
    # Create datasets from easy_train, train and valid labels
    ds_easy_train_labels = create_labels_dataset(easy_train_labels)  # Easy scenes
    ds_train_labels = create_labels_dataset(train_labels)  # All images
    ds_valid_labels = create_labels_dataset(valid_labels)  # Valid images

    # Get number of images
    n_easy_train_images = len(list_easy_train_images_path)
    n_train_images = len(list_train_images_path)
    n_valid_images = len(list_valid_images_path)

    # Zip all data (images, labels)
    ds_easy_train_data = tf.data.Dataset.zip(ds_easy_train_images, ds_easy_train_labels).\
        shuffle(n_easy_train_images,reshuffle_each_iteration=True, seed=config.SEED).cache().\
        map(data_augmentation, num_parallel_calls=tf.data.AUTOTUNE).repeat().\
        batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    ds_train_data = tf.data.Dataset.zip(ds_train_images, ds_train_labels).\
        shuffle(n_train_images,reshuffle_each_iteration=True, seed=config.SEED).cache().\
        map(data_augmentation,num_parallel_calls=tf.data.AUTOTUNE).repeat().\
        batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    ds_valid_data = tf.data.Dataset.zip(ds_valid_images, ds_valid_labels).cache().repeat().\
        batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    # Create test dataset
    ds_test_data = ds_test_images.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)


    # -------------------------------------------------
    # 7. MODEL ARCHITECTURE
    # -------------------------------------------------
    print('---- Model construction ----')
    # Models construction
    backbone = build_backbone()
    input_shape = backbone.output.shape[-1]
    model = Model(base_model=backbone,
                  quat_head=create_quat_head(input_shape, config.DROPOUT_RATE, config.L2_REG),
                  trans_head=create_trans_head(input_shape, config.DROPOUT_RATE, config.L2_REG),
                  outlier_head=create_outl_head(input_shape, config.DROPOUT_RATE, config.L2_REG),
                  trans_mean=transl_scaler.mean_,
                  trans_scale=transl_scaler.scale_,
                  batch_size=batch_size)

    # Feature extractor model construction
    feature_extractor = tf.keras.Model(inputs=backbone.input, outputs=backbone.output)


    # -------------------------------------------------
    # 8. TRAINING PARAMETERS
    # -------------------------------------------------
    print('---- Preparation for training step ----')
    # Definition of Early_Stopping callback
    early_stopping = early_stopping_callback()

    # Definition of CosineDecay scheduler for Learning Rate in each phase
    # Training with warm-up
    steps_per_epochs = math.ceil(n_train_images / batch_size)
    num_steps_training = config.NUM_EPOCHS * steps_per_epochs
    num_steps_warmup = round(num_steps_training * config.WARMUP_STEPS_RATE)
    print(f'Number of warm-up steps: {num_steps_warmup}')

    lr_warmup_decayed = learning_rate_cosine_decay(initial_lr=config.LR_INITIAL_WARMUP,
                                                   decay_steps=num_steps_training - num_steps_warmup,
                                                   warmup_lr_target=config.LR_TARGET_WARMUP,
                                                   warmup_steps=num_steps_warmup)

    lr_all_images = learning_rate_cosine_decay(initial_lr=config.LR_INITIAL,
                                               decay_steps=num_steps_training,
                                               warmup_lr_target=None,
                                               warmup_steps=0)

    # Fine-tuning (phase 1)
    num_steps_ft1 = config.NUM_EPOCHS_FT1 * steps_per_epochs
    lr_ft_1 = learning_rate_cosine_decay(initial_lr=config.LR_INITIAL_FT1,
                                         decay_steps=num_steps_ft1,
                                         warmup_lr_target=None,
                                         warmup_steps=0)

    # Fine-tuning (phase 2)
    num_steps_ft2 = config.NUM_EPOCHS_FT2 * steps_per_epochs
    lr_ft_2 = learning_rate_cosine_decay(initial_lr=config.LR_INITIAL_FT2,
                                         decay_steps=num_steps_ft2,
                                         warmup_lr_target=None,
                                         warmup_steps=0)


    # -------------------------------------------------
    # 9. MODEL CONSTRUCTION AND TRAINING
    # -------------------------------------------------
    print('---- Train model ----')
    validation_steps = n_valid_images // batch_size
    df_history = training_model(strategy=strategy,
                                model=model,
                                backbone=backbone,
                                ds_easy_train_data=ds_easy_train_data,
                                ds_train_data=ds_train_data,
                                ds_valid_data=ds_valid_data,
                                lr_warmup_decayed=lr_warmup_decayed,
                                lr_all_images=lr_all_images,
                                lr_ft_1=lr_ft_1,
                                lr_ft_2=lr_ft_2,
                                early_stopping=early_stopping,
                                steps_per_epochs=steps_per_epochs,
                                validation_steps=validation_steps,
                                config=config)


    # -------------------------------------------------
    # 10. PLOTS OF TRAINING LOGS
    # -------------------------------------------------
    print('---- Saving plots from training ----')
    # Loss plot
    plot_results(df=df_history,
                 list_values=['loss', 'val_loss'],
                 filename='loss_curves.png',
                 config=config)

    # Quaternion lost plot
    plot_results(df=df_history,
                 list_values=['quaternions_loss', 'quaternions_cos_similarity',
                              'val_quaternions_loss', 'val_quaternions_cos_similarity'],
                 filename='loss_quaternions.png',
                 config=config)

    # Traslation lost plot
    plot_results(df=df_history,
                 list_values=['translations_loss','translations_mae',
                              'val_translations_loss','val_translations_mae'],
                 filename='loss_translations.png',
                 config=config)

    # Outliers lost plot
    plot_results(df=df_history,
                 list_values=['outliers_loss', 'outliers_binary_accuracy',
                              'val_outliers_loss', 'val_outliers_binary_accuracy'],
                 filename='loss_outliers.png',
                 config=config)


    # -------------------------------------------------
    # 11. INFERENCE
    # -------------------------------------------------
    print('---- Obtaining predictions ----')
    # Get predictions
    visual_features, predictions = run_inference(feature_extractor=feature_extractor,
                                                 model=model,
                                                 test_dataset=ds_test_data,  # Dataset with only the images
                                                 path=config)

    # Save predictions
    np.save(config.OUTPUT_PATH / 'visual_features_pred.npy', visual_features)
    np.save(config.OUTPUT_PATH / 'predictions.npy', predictions)


    # -------------------------------------------------
    # 12. CLUSTERING IMAGES
    # -------------------------------------------------
    # Get clusters
    threshold_rot = stats['rot_pct95'].mean()  # Mean of 95-percentile of rotations of all scenes
    threshold_trans = np.median(train_thresholds_file['thresholds']
                                .apply(lambda x: min(x)).values)  # Median of lowest values of threshold of all scenes
    labels = clusters_prediction(predictions, visual_features, threshold_rot, threshold_trans)


    # -------------------------------------------------
    # 16. OUTPUT SUBMISSION FILE
    # -------------------------------------------------
    sample_submission = create_submission_file(config.TEST_FOLDER, list_test_images_path, labels, predictions)


    # -------------------------------------------------
    # 17. PLOT CLUSTERS
    # -------------------------------------------------
    # - Functions definition - #
    # Plot images
    print_outliers = False
    for i in range(6):
        # Define number of cluster
        n_cluster = 'cluster' + str(i)
        images_cluster = sample_submission[sample_submission['scene'] == n_cluster][['dataset', 'image']]
        images_cluster = list(config.TEST_FOLDER / images_cluster['dataset'] / images_cluster['image'])

        # Plot each cluster
        if len(images_cluster) > 0:
            plot_clusters(images_cluster, n_cluster)

        # Plot outliers (after plotting all clusters)
        elif not print_outliers:
            n_cluster = 'outliers'
            images_cluster = sample_submission[sample_submission['scene'] == n_cluster][['dataset', 'image']]
            images_cluster = list(config.TEST_FOLDER / images_cluster['dataset'] / images_cluster['image'])
            if len(images_cluster) > 0:
                plot_clusters(images_cluster, n_cluster)
                print_outliers = True

        # Nothing to plot (after plotting all clusters and outliers)
        else:
            pass


if __name__ == '__main__':
    main()
