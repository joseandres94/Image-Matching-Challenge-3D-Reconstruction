from src import config

import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping


# Define callbacks
def early_stopping_callback():
    return EarlyStopping(monitor='val_loss', patience=config.ES_PATIENCE, restore_best_weights=True)


def learning_rate_cosine_decay(initial_lr, decay_steps, warmup_lr_target, warmup_steps):
    return tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=initial_lr,
                                                     decay_steps=decay_steps,
                                                     warmup_target=warmup_lr_target,
                                                     warmup_steps=warmup_steps,
                                                     alpha=0)


# Model compilation
def compile_model(model, optimizer, use_tpu):
    steps = 30 if use_tpu else 1
    model.compile(optimizer=optimizer,
                  steps_per_execution=steps,
                  run_eagerly=False)


# Training model
def training_model(strategy, model, backbone, ds_easy_train_data, ds_train_data, ds_valid_data, lr_warmup_decayed,
                   lr_all_images, lr_ft_1, lr_ft_2, early_stopping, steps_per_epochs, validation_steps,
                   config):
    # Full model construction and training
    with strategy.scope():
        use_tpu = isinstance(strategy, tf.distribute.TPUStrategy)

        # Model compilation
        adam = tf.keras.mixed_precision.LossScaleOptimizer(
            tf.optimizers.Adam(learning_rate=lr_warmup_decayed)
        )
        compile_model(model, adam, use_tpu)

        # Train model (with warm-up)
        # Only easy scenes
        print('---- Training model (on easy scenes) ----')
        history_t1 = model.fit(x=ds_easy_train_data,
                               validation_data=ds_valid_data,
                               epochs=config.NUM_EPOCHS,
                               steps_per_epoch=steps_per_epochs,
                               validation_steps=validation_steps,
                               callbacks=[early_stopping])

        # Refresh number epochs completed
        num_epochs_completed = len(history_t1.epoch)

        # All images
        # Model compilation
        adam = tf.keras.mixed_precision.LossScaleOptimizer(
            tf.optimizers.Adam(learning_rate=lr_all_images)
        )
        compile_model(model, adam, use_tpu)

        print('---- Training model (on all images) ----')
        history_t2 = model.fit(x=ds_train_data,
                               validation_data=ds_valid_data,
                               epochs=num_epochs_completed + config.NUM_EPOCHS,
                               steps_per_epoch=steps_per_epochs,
                               validation_steps=validation_steps,
                               callbacks=[early_stopping],
                               initial_epoch=num_epochs_completed)

        # Refresh number epochs completed
        num_epochs_completed += len(history_t2.epoch)

        # Fine-tuning
        print('---- Fine-tuning model (Phase 1) ----')
        # Set % first layers not trainable
        for layer in backbone.layers[int(len(backbone.layers) * config.FROZEN_LAYERS_RATE_1):]:
            layer.trainable = True

        # Model compilation
        adam_ft = tf.keras.mixed_precision.LossScaleOptimizer(
            tf.optimizers.Adam(learning_rate=lr_ft_1)
        )
        compile_model(model, adam_ft, use_tpu)

        # Fine-tuning model
        history_ft_1 = model.fit(x=ds_train_data,
                                 validation_data=ds_valid_data,
                                 epochs=num_epochs_completed + config.NUM_EPOCHS_FT1,
                                 steps_per_epoch=steps_per_epochs,
                                 validation_steps=validation_steps,
                                 callbacks=[early_stopping],
                                 initial_epoch=num_epochs_completed)

        # Refresh number epochs completed
        num_epochs_completed += len(history_ft_1.epoch)

        print('---- Fine-tuning model (Phase 2) ----')
        # Set % first layers not trainable
        for layer in backbone.layers[int(len(backbone.layers) * config.FROZEN_LAYERS_RATE_2):]:
            layer.trainable = True

        # Model compilation
        adam_ft = tf.keras.mixed_precision.LossScaleOptimizer(
            tf.optimizers.Adam(learning_rate=lr_ft_2)
        )
        compile_model(model, adam_ft, use_tpu)

        # Fine-tuning model
        history_ft_2 = model.fit(x=ds_train_data,
                                 validation_data=ds_valid_data,
                                 epochs=num_epochs_completed + config.NUM_EPOCHS_FT2,
                                 steps_per_epoch=steps_per_epochs,
                                 validation_steps=validation_steps,
                                 callbacks=[early_stopping],
                                 initial_epoch=num_epochs_completed)

        print('---- Training finished ----')

        # Convert history data to DataFrame
        df_history_t1 = pd.DataFrame(history_t1.history)
        df_history_t1['epoch'] = history_t1.epoch

        df_history_t2 = pd.DataFrame(history_t2.history)
        df_history_t2['epoch'] = history_t2.epoch

        df_history_ft_1 = pd.DataFrame(history_ft_1.history)
        df_history_ft_1['epoch'] = history_ft_1.epoch

        df_history_ft_2 = pd.DataFrame(history_ft_2.history)
        df_history_ft_2['epoch'] = history_ft_2.epoch

        df_history = pd.concat([df_history_t1, df_history_t2, df_history_ft_1, df_history_ft_2])
        df_history.set_index('epoch', inplace=True)

        # Save models
        print('---- Saving data----')
        model.export(config.OUTPUT_PATH / 'model')
        backbone.export(config.OUTPUT_PATH / 'feature_extractor')

        # Save history data
        df_history.to_csv(config.OUTPUT_PATH / 'training_history.csv', index=False)

        print('---- Process finished ----')

        return df_history
