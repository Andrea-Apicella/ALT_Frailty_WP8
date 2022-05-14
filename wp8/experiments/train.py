import numpy as np
import pandas as pd
import tensorflow as tf
import wandb
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tqdm.notebook import tqdm
from wandb.keras import WandbCallback
from wp8.pre_processing.generators import TimeSeriesGenerator as TSG
from wp8.pre_processing.utils import listdir_nohidden_sorted as lsdir

#RANDOM SEEDS
np.random.seed(2)
tf.random.set_seed(2)

#WANDB LOGIN
%env WANDB_API_KEY =$a22c5c63cb14ecd62db2141ec9ca69d588a6483e
wandb.login()

##PATHS
features_path = "../outputs/dataset/features/"
dataset_path = "../outputs/dataset/dataset/"

#load features
all_features = []
all_features_paths = lsdir(features_path)
