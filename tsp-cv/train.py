import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import zipfile
import wandb
import tensorflow as tf
import cv2

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error
os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "50"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

tf.random.set_seed(42)
tf.config.run_functions_eagerly(False)

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

PATH = "dataset/new/"
PATH_TRAIN = os.path.join(PATH, "train.pkl")
PATH_TEST = os.path.join(PATH, "test.pkl")

train_df = pd.read_pickle(PATH_TRAIN)
test_df = pd.read_pickle(PATH_TEST)

X_train, X_test, y_train, y_test = train_test_split(train_df["image_hist"].to_list(), train_df["distance"].values,
                                                    test_size=0.15, train_size=0.85, random_state=42)


def prepare(ds, shuffle=False, repeat=False, cache=False, batch_size=32):

    if shuffle:
        ds = ds.shuffle(buffer_size=1000)

    if repeat:
        ds = ds.repeat()

    ds = ds.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        ds = ds.cache()

    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)


def Model(fc_layer):

    model = models.Sequential([

        layers.Input(shape=(3, 256,), name="input_layer", dtype=tf.float32),
        layers.Flatten(),
        layers.Dense(fc_layer, activation="relu"),
        layers.Dense(fc_layer/wandb.config.ratio, activation="relu"),
        layers.Dense(1, activation="linear",
                     name="output_layer", dtype=tf.float32)

    ])

    return model


def calculate_results(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    max_err = max_error(y_true, y_pred)

    wandb.log({"valid_mae": mae,
               "valid_mse": mse,
               "valid_rmse": np.sqrt(mse),
               "valid_err": max_err})


def train():

    wandb.init(project="sweep-tsp")
    # build input pipeline using tf.data
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_data = prepare(train_data, batch_size=wandb.config.batch_size, cache=True)

    valid_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    valid_data = prepare(valid_data, batch_size=wandb.config.batch_size, cache=True)

    # initialize model
    model = Model(wandb.config.fc_layer_size)
    model.compile(optimizer=Adam(learning_rate=wandb.config.learning_rate),
                  loss="mse",
                  metrics=[RootMeanSquaredError(name="rmse")])

    stop_callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4,
                                        patience=25, verbose=1, mode='auto',
                                        restore_best_weights=True)

    lr_callback = callbacks.ReduceLROnPlateau(monitor='loss',
                                          factor=0.2, min_lr=1e-10, patience=2)
    history = (
        model.fit(
            train_data,
            epochs=2000,
            validation_data=valid_data,
            callbacks=[stop_callback, lr_callback]
        )
    )

    model_pred = model.predict(valid_data)
    calculate_results(y_test, model_pred)

    del model
    tf.keras.backend.clear_session()


if __name__ == '__main__':
    train()
