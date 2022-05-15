# Evaluating CNN+RNN models on the dataset

# Imports
from datetime import datetime
from statistics import mode

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
# from sklearn.metrics import classification_report
# from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.metrics import Accuracy, SparseCategoricalAccuracy
from tensorflow.keras.models import Sequential
from tqdm.notebook import tqdm
from wandb.keras import WandbCallback
from wp8.pre_processing.generators import TimeSeriesGenerator as TSG
from wp8.pre_processing.utils import listdir_nohidden_sorted as lsdir
from wp8.pre_processing.utils import safe_mkdir

import wandb

# Set random seeds
np.random.seed(2)
tf.random.set_seed(2)


# wandb.login()
# %env WANDB_API_KEY=$a22c5c63cb14ecd62db2141ec9ca69d588a6483e


# Load dataset and features
features_path = "../outputs/dataset/features/"
dataset_path = "../outputs/dataset/dataset/"

# load features
all_features = []
all_features_paths = lsdir(features_path)[0:1]
for _, feature_file in enumerate(tqdm(all_features_paths)):
    with np.load(feature_file) as features:
        all_features.append(features["arr_0"])

all_features = np.concatenate(all_features, axis=0)


dfs = []
for _, filename in enumerate(tqdm(lsdir(dataset_path)[0:1])):
    df = pd.read_csv(filename, index_col=0)
    dfs.append(df)

dataset = pd.concat(dfs, ignore_index=True)

print(f"dataset shape: {dataset.shape}, all_features shape: {all_features.shape}")

dataset.head(-10)

names = dataset["frame_name"]
cams = []
for name in names:
    cams.append(int(name[-6]))

dataset["cams"] = pd.Series(cams)

dataset.head()


# insert features in the dataframe
dataset["features"] = pd.Series(all_features.tolist())

# count samples per label, get labels names, encode labels to integers
dataset["micro_labels"].value_counts()
micro_labels_names = dataset["micro_labels"].unique().tolist()

le = preprocessing.LabelEncoder()
encoded_labels = le.fit_transform(dataset["micro_labels"])
n_labels = len(np.unique(encoded_labels))


# WANDB project initialization
run = wandb.init(
    project="Fall detection CNN + RNN",
    config={
        "model": "LSTM",
        "epochs": 5,
        "sequence_length": 20,
        "num_features": 2048,
        "batch_size": 40,
        "sliding_window_stride": 10,
        "loss_function": "sparse_categorical_crossentropy",
        "architecture": "LSTM",
        "dataset": "Actor_1_Bed",
        "dropout": 0.8,
        "lstm1_units": 32,
        "learning_rate": 0.01,
        "split_ratio": 0.7,
    },
)

config = wandb.config


# Train Test split
split = int(dataset.shape[0] * config.split_ratio)
X_train = np.array(dataset["features"][0:split].tolist())
X_test = np.array(dataset["features"][split:].tolist())

y_train = encoded_labels[0:split]
y_test = encoded_labels[split:]

cams_train = dataset["cams"][0:split]
cams_test = dataset["cams"][split:]

print(f"X_train shape :{X_train.shape}, y_train shape: {y_train.shape}, X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


print(f'Last train frame: {dataset["frame_name"][split]}\nFirst test frame: {dataset["frame_name"][split+1]}')


# Create Model
train_gen = TSG(
    X=X_train,
    y=y_train,
    num_features=config.num_features,
    cams=cams_train.tolist(),
    batch_size=config.batch_size,
    stride=config.sliding_window_stride,
    seq_len=config.sequence_length,
)
test_gen = TSG(
    X=X_test,
    y=y_test,
    cams=cams_test.tolist(),
    num_features=config.num_features,
    batch_size=config.batch_size,
    stride=config.sliding_window_stride,
    seq_len=config.sequence_length,
)

model = Sequential()
model.add(LSTM(units=config.lstm1_units, input_shape=(20, config.num_features)))
model.add(Dropout(config.dropout))
model.add(Dense(n_labels, activation="softmax"))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
    loss=config.loss_function,
    metrics=["accuracy", "sparse_categorical_accuracy"],
)
model.summary()


# Callbacks
dir_path = f"experiments/model_checkpoint/{config.model}_{config.dataset}"
safe_mkdir(dir_path)
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
model_checkpoint = ModelCheckpoint(
    filepath=f"{dir_path}/{config.model}_{dt_string}",
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
)
callbacks = [WandbCallback(), model_checkpoint]


# Train Model
history = model.fit(train_gen, validation_data=test_gen, epochs=config.epochs, callbacks=callbacks)
test_gen.evaluate = True


# Evaluate Model
test_logits = model.predict_generator(test_gen, verbose=1)

print(
    f"test_gen.n_windows: {test_gen.n_windows}\n\ntest_gen series_labels length: {len(test_gen.series_labels)}\n\nCorrect number of labels: {test_gen.n_windows * (y_test.shape[0] // test_gen.batch_size)}\n\nlogits shape: {test_logits.shape}"
)


def to_series_labels(timestep_labels: list, n_batches: int, n_windows: int, seq_len: int, stride: int) -> list:
    series_labels = []
    for w in range(n_windows * n_batches):
        s = w * stride
        labels_seq = timestep_labels[s : s + seq_len]
        series_labels.append(mode(labels_seq))
    return series_labels


# Log metrics to wandb

y_pred_test_classes = np.argmax(test_logits, axis=1).tolist()
y_train_series = to_series_labels(
    y_train,
    train_gen.n_batches,
    train_gen.n_windows,
    train_gen.seq_len,
    train_gen.stride,
)
y_test_series = to_series_labels(y_test, test_gen.n_batches, test_gen.n_windows, test_gen.seq_len, test_gen.stride)
wandb.sklearn.plot_roc(y_test_series, test_logits, micro_labels_names)
wandb.sklearn.plot_class_proportions(y_train_series, y_test_series, micro_labels_names)
wandb.sklearn.plot_precision_recall(y_test_series, test_logits, micro_labels_names)
wandb.sklearn.plot_confusion_matrix(y_test_series, y_pred_test_classes, micro_labels_names)
wandb.join()
