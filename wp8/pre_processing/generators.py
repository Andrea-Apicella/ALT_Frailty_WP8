from cProfile import label
from statistics import mode

import numpy as np
from tensorflow.keras.utils import Sequence


class TimeSeriesGenerator(Sequence):
    def __init__(self, X: np.ndarray, y: list, num_features: int, cams: list, seq_len: int, stride: int, batch_size: int, labels_encoder, evaluate=False):
        self.X = X
        self.y = y
        self.num_features = num_features
        self.cams = cams
        self.seq_len = seq_len
        self.stride = stride
        self.batch_size = batch_size
        self.n = len(self.X)
        self.count = 1
        self.len = len(self.X)
        self.evaluate = evaluate
        self.n_windows = (self.batch_size - self.seq_len) // self.stride + 1
        self.n_series_labels = self.n_windows * (len(self.y) // self.batch_size)
        self.n_batches = len(self.y) // self.batch_size
        self.series_labels = []
        self.ys_count = 0
        self.get_item_calls = 0
        self.labels_encoder = labels_encoder

    def __len__(self):
        return self.X.shape[0] // (self.batch_size)

    def __get_data(self, batch):
        # generate one time series of seq_len padded if its too short
        # check that the time series is from the one cam only (does not overflow on another camera)
        X = batch["features"]
        y = batch["labels"]
        cams = batch["cams"]

        time_series = [np.empty(self.num_features)] * self.n_windows
        y_s = [None] * self.n_windows
        s = 0
        for w in range(0, self.n_windows):
            s = w * self.stride
            features_seq = X[s : s + self.seq_len, :]
            labels_seq = y[s : s + self.seq_len]
            cams_seq = cams[s : s + self.seq_len]
            curr_cam = mode(cams_seq)
            for i, _ in enumerate(cams_seq):
                if cams_seq[i] != curr_cam:
                    features_seq[i] = np.zeros(self.num_features)  # padding
                    labels_seq[i] = -10  # padding
            time_series[w] = features_seq
            # convert time-step labels in one label per time-series
            labels_seq = [l for l in labels_seq if l != -10]
            label = mode(labels_seq)  # label with most occurrence
            y_s[w] = label

        if not self.evaluate:
            self.series_labels.extend(y_s)
            self.ys_count += len(y_s)
        return np.array(time_series), self.labels_encoder.fit_transform(y_s)

    def __getitem__(self, index):
        self.get_item_calls += 1
        a = index * self.batch_size
        b = (index + 1) * self.batch_size

        batch = {"features": self.X[a:b, :], "labels": self.y[a:b], "cams": self.cams[a:b]}
        X, y = self.__get_data(batch)
        return X, y

    def __on_epoch_end(self):
        pass
