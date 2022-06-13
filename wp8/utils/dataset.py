from statistics import mode

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from this import d
from tqdm.notebook import trange
from collections import Counter


class TimeSeriesGenerator:
    def __init__(self, opt):
        self.labels_encoder = LabelEncoder()
        self.n_windows: int = (opt.batch_size - opt.seq_len) // opt.stride + 1
        self.seq_len = opt.seq_len
        self.batch_size = opt.batch_size
        self.stride = opt.stride
        self.num_cnn_features = 2048

    def __to_time_series(self, X: np.ndarray, y: list, cams: list, n_series: int) -> tuple[list, list]:
        time_series: list = [np.empty(self.num_cnn_features)] * n_series
        y_s: list = [None] * n_series
        s: int = 0
        for w in trange(n_series):
            s = w * self.stride
            features_seq = X[s : s + self.seq_len, :]
            labels_seq = y[s : s + self.seq_len]
            cams_seq = cams[s : s + self.seq_len]
            curr_cam = mode(cams_seq)
            for i, _ in enumerate(cams_seq):
                if cams_seq[i] != curr_cam:
                    features_seq[i] = np.zeros(self.num_cnn_features)  # padding
                    labels_seq[i] = -10  # padding
            time_series[w] = features_seq
            # convert time-step labels in one label per time-series
            labels_seq = [l for l in labels_seq if l != -10]
            label = mode(labels_seq)  # label with most occurrence
            y_s[w] = label
        return time_series, y_s

    def get_train_series(self, X: np.ndarray, y: list, cams: list) -> tuple[np.ndarray, np.ndarray, dict, list]:
        n_series: int = (len(y) - self.seq_len) // self.stride + 1

        X_series, y_series = self.__to_time_series(X, y, cams, n_series)
        distrib = Counter(y_series)
        self.labels_encoder = self.labels_encoder.fit(y_series)
        mapping = dict(zip(self.labels_encoder.classes_, range(1, len(self.labels_encoder.classes_) + 1)))
        y_series = self.labels_encoder.fit_transform(y_series)

        print(f"Classes mapping:\n{mapping}")
        print(f"\nClass distribution:\n{distrib}")

        return np.array(X_series), y_series, distrib, self.labels_encoder.classes_.tolist()

    def get_val_series(self, X: np.ndarray, y: list, cams: list):
        n_series: int = (len(y) - self.seq_len) // self.stride + 1

        X_series, y_series = self.__to_time_series(X, y, cams, n_series)

        return np.array(X_series), self.labels_encoder.fit_transform(y_series)
