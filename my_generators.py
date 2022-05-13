from statistics import mode
from typing import Sequence

import numpy as np
from tensorflow.keras.utils import Sequence


class TimeSeriesGenerator(Sequence):
    def __init__(self, X, y: list, num_features: int, cams: list, seq_len: int, stride: int, batch_size: int):
        self.X = X
        self.y = y.tolist()
        self.num_features = num_features
        self.cams = cams
        self.seq_len = seq_len
        self.stride = stride
        self.batch_size = batch_size
        self.n = len(self.X)
        self.count = 1
        self.len = len(self.X)

    def __len__(self):
        return self.X.shape[0] // self.batch_size

    # def __pad(self, series, value):
    #     # pad time series if it is too short
    #     if len(series.shape[1] < self.seq_len):
    #         # np.pad(series)
    #         return series

    def __get_data(self, batch):
        # generate one time series of seq_len padded if its too short
        # check that the time series is from the one cam only (does not overflow on another camera)
        X = batch["features"]
        y = batch["labels"]
        cams = batch["cams"]
        time_series = []
        y_s = []
        s = 0
        while s + self.seq_len <= (X.shape[0]):
            features_seq = X[s:s+self.seq_len]
            labels_seq = y[s:s+self.seq_len]
            cams_seq = cams[s:s+self.seq_len]
            curr_cam = mode(cams_seq)
            for i, _ in enumerate(cams_seq):
                if cams_seq[i] != curr_cam:
                    features_seq[i] = np.zeros(self.num_features)  # padding
                    labels_seq[i] = -1  # padding
            time_series.append(features_seq)
            labels_seq = filter(lambda label: label != -1, labels_seq)
            label = mode(labels_seq)  # most occurrent label
            y_s.append(label)
            s += self.stride
            # print(
            #     f"np.array(time_series.shape) {np.array(time_series).shape}")
            # print(f"np.array(y_s).shape {np.array(y_s).shape}")
        return np.asarray(time_series), np.asarray(y_s)

    def __getitem__(self, index):
        print(index)
        a = index * self.batch_size
        b = (index+1) * self.batch_size * self.seq_len

        batch = {"features": self.X[a:b+1],
                 "labels": self.y[a:b+1], "cams": self.cams[a:b+1]}
        X, y = self.__get_data(batch)
        print(f"X shape: {X.shape}, y shape:{y.shape}")
        return X, y

    def __on_epoch_end(self):
        pass
