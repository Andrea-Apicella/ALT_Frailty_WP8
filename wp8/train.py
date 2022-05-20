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
from sklearn import preprocessing
from sklearn.utils.class_weight import compute_class_weight

# from sklearn.metrics import classification_report
# from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from wandb.keras import WandbCallback

import wandb
from wp8.options.train_options import TrainOptions
from wp8.pre_processing.generators import TimeSeriesGenerator as TSG
from wp8.pre_processing.utils import safe_mkdir
from wp8.utils.cnn_rnn_utils import load_and_split, to_series_labels

# Set random seeds
np.random.seed(2)
tf.random.set_seed(2)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


opt = TrainOptions().parse()

# check that actors and cams passed to the parsers are not in common between train and val sets
if set(opt.train_actors) & set(opt.val_actors):
    raise Exception("Can't use the same actors both in train and validation splits")

# opt.train_actors
# WANDB project initialization
run = wandb.init(
    project="Fall detection CNN + RNN",
    config={
        "model": "LSTM",
        "epochs": opt.epochs,
        "sequence_length": opt.seq_len,
        "num_features": 2048,
        "batch_size": opt.batch_size,
        "sliding_window_stride": opt.stride,
        "loss_function": "sparse_categorical_crossentropy",
        "architecture": "LSTM",
        "train_actors": opt.train_actors,
        "val_actors": opt.val_actors,
        "train_cams": opt.train_cams,
        "val_cams": opt.val_cams,
        "dropout": opt.dropout,
        "lstm1_units": opt.lstm1_units,
        "lstm2_units": opt.lstm2_units,
        "learning_rate": opt.learning_rate,
        "split_ratio": opt.split_ratio,
        "drop_offair": opt.drop_offair,
        "undersample": opt.undersample,
    },
)

cfg = wandb.config


X_train, y_train, X_val, y_val, cams_train, cams_val, classes = load_and_split(opt.train_actors, opt.val_actors, opt.train_cams, opt.val_cams, opt.split_ratio, opt.drop_offair, opt.undersample)


print(f"\nX_train shape: {X_train.shape}, len y_train: {len(y_train)}, X_val shape: {X_val.shape}, len y_val: {len(y_val)}\n")

# Create Model
train_gen = TSG(
    X=X_train,
    y=y_train,
    num_features=cfg.num_features,
    cams=cams_train,
    batch_size=cfg.batch_size,
    stride=cfg.sliding_window_stride,
    seq_len=cfg.sequence_length,
)
val_gen = TSG(
    X=X_val,
    y=y_val,
    cams=cams_val,
    num_features=cfg.num_features,
    batch_size=cfg.batch_size,
    stride=cfg.sliding_window_stride,
    seq_len=cfg.sequence_length,
)

y_train_series = to_series_labels(y_train, train_gen.n_batches, train_gen.n_windows, train_gen.seq_len, train_gen.stride)
y_val_series = to_series_labels(y_val, val_gen.n_batches, val_gen.n_windows, val_gen.seq_len, val_gen.stride)
y_train_series_unique = np.unique(y_train_series)
y_val_series_unique = np.unique(y_val_series)

if y_train_series_unique.sort() != y_val_series_unique.sort():
    raise Exception("y_train_series_unique != y_val_series_unique")


model = Sequential()
model.add(LSTM(units=cfg.lstm1_units, input_shape=(20, cfg.num_features), return_sequences=True))
model.add(Dropout(cfg.dropout))
model.add(LSTM(units=cfg.lstm2_units, input_shape=(20, cfg.num_features)))
model.add(Dropout(cfg.dropout))
model.add(Dense(len(y_train_series_unique), activation="softmax"))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
    loss=cfg.loss_function,
    metrics=["accuracy", "sparse_categorical_accuracy"],
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

callbacks = [WandbCallback(), model_checkpoint]

# Class Weights
class_weights = compute_class_weight(class_weight="balanced", classes=y_train_series_unique, y=y_train_series)
class_weights = dict(zip(y_train_series_unique, class_weights))
print(f"\nClasses mapping: {classes}")
print(f"\nClass weights for train series: {class_weights}")

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