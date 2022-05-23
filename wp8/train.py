# Evaluating CNN+RNN models on the dataset

# Imports
import csv
import gc
import os
from collections import Counter
from datetime import datetime
from statistics import mode

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.activations import ELU
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from wandb.keras import WandbCallback

import wandb
from wp8.options.train_options import TrainOptions
from wp8.pre_processing.generators import TimeSeriesGenerator as TSG
from wp8.pre_processing.utils import safe_mkdir
from wp8.utils.cnn_rnn_utils import (get_timeseries_labels_encoded,
                                     load_and_split)

opt = TrainOptions().parse()


if set(opt.train_actors) & set(opt.val_actors):
    raise Exception("Can't use the same actors both in train and validation splits")


# WANDB project initialization
run = wandb.init(
    project="Fall detection CNN + RNN",
    config={
        "model": "LSTM",
        "epochs": opt.epochs,
        "seq_len": opt.seq_len,
        "num_features": 2048,
        "batch_size": opt.batch_size,
        "stride": opt.stride,
        "loss_function": "sparse_categorical_crossentropy",
        "architecture": "LSTM",
        "train_actors": opt.train_actors,
        "val_actors": opt.val_actors,
        "train_cams": opt.train_cams,
        "val_cams": opt.val_cams,
        "dropout": opt.dropout,
        "lstm1_units": opt.lstm1_units,
        "lstm2_units": opt.lstm2_units,
        "dense_units": opt.dense_units,
        "learning_rate": opt.learning_rate,
        "split_ratio": opt.split_ratio,
        "drop_offair": opt.drop_offair,
        "undersample": opt.undersample,
    },
)

cfg = wandb.config


X_train, y_train, X_val, y_val, cams_train, cams_val = load_and_split(opt.train_actors, opt.val_actors, opt.train_cams, opt.val_cams, opt.split_ratio, opt.drop_offair, opt.undersample)
print(f"\nX_train shape: {X_train.shape}, len y_train: {len(y_train)}, X_val shape: {X_val.shape}, len y_val: {len(y_val)}\n")


y_train_series, y_val_series, enc, class_weights, classes = get_timeseries_labels_encoded(y_train, y_val, cfg)

print(f"Classes: {classes}\nClass weights: {class_weights}")


train_gen = TSG(
    X=X_train,
    y=y_train,
    num_features=cfg.num_features,
    cams=cams_train,
    batch_size=cfg.batch_size,
    stride=cfg.stride,
    seq_len=cfg.seq_len,
    labels_encoder=enc,
)
val_gen = TSG(
    X=X_val,
    y=y_val,
    cams=cams_val,
    num_features=cfg.num_features,
    batch_size=cfg.batch_size,
    stride=cfg.stride,
    seq_len=cfg.seq_len,
    labels_encoder=enc,
)


model = Sequential()
model.add(LSTM(units=cfg.lstm1_units, input_shape=(cfg.seq_len, cfg.num_features), return_sequences=True))
model.add(Dropout(cfg.dropout))
model.add(LSTM(units=cfg.lstm2_units, input_shape=(cfg.seq_len, cfg.num_features)))
model.add(Dropout(cfg.dropout))
model.add(Dense(units=cfg.dense_units))
model.add(ELU())
model.add(Dropout(cfg.dropout))
model.add(Dense(units=np.unique(y_train_series, axis=0).shape[0], activation="softmax"))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
    loss=cfg.loss_function,
    metrics=["accuracy", "sparse_categorical_crossentropy"],
)
model.summary()


# Callbacks
dir_path = f"model_checkpoints/{cfg.model}"
safe_mkdir(dir_path)
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")

model_checkpoint = ModelCheckpoint(
    filepath=f"{dir_path}/{cfg.model}_{dt_string}",
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
    initial_value_threshold=0.8,
    verbose=1,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.1,
    patience=10,
    verbose=1,
    mode="auto",
    min_delta=1e-5,
    cooldown=1,
    min_lr=1e-9,
)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    min_delta=0,
    patience=10,
    verbose=1,
    mode="auto",
)

callbacks = [WandbCallback(), model_checkpoint, reduce_lr, early_stop]


# Train Model
history = model.fit(train_gen, validation_data=val_gen, epochs=cfg.epochs, callbacks=callbacks, class_weight=class_weights)
val_gen.evaluate = True


# Evaluate Model
val_logits = model.predict(val_gen, verbose=1)


# free up memory
del X_train
del y_train
del X_val
del y_val

gc.collect()


# Log metrics to wandb
y_pred_val_classes = np.argmax(val_logits, axis=1).tolist()

wandb.sklearn.plot_roc(y_val_series, val_logits, classes)
wandb.sklearn.plot_class_proportions(y_train_series, y_val_series, classes)
wandb.sklearn.plot_precision_recall(y_val_series, val_logits, classes)
wandb.sklearn.plot_confusion_matrix(y_val_series, y_pred_val_classes, classes)
wandb.join()
