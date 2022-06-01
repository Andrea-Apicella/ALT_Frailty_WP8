from collections import Counter
from statistics import mode

import numpy as np
import pandas as pd
from imblearn.under_sampling import NearMiss
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm, trange
from wp8.pre_processing.utils import listdir_nohidden_sorted as lsdir


class DatasetLoader:
    def __init__(self, dataset_folder: str, features_folder: str, actors: list, cams: list, drop_offair: bool):
        self.dataset_folder = dataset_folder
        self.features_folder = features_folder
        self.actors = actors
        self.cams = cams
        self.drop_offair = drop_offair

    def load(self) -> tuple[pd.Series, np.ndarray]:
        datasets_paths = lsdir(self.dataset_folder)
        features_paths = lsdir(self.features_folder)

        if self.actors:
            self.actors = [int(a) for a in self.actors]
            indexes = []
            for i, filename in enumerate(datasets_paths):
                if int(filename[filename.find("Actor_") + 6]) not in self.actors:
                    indexes.append(i)

            for index in sorted(indexes, reverse=True):
                del datasets_paths[index]
                del features_paths[index]

        # load features
        all_features = []
        for _, feature_file_name in enumerate((t := tqdm(features_paths))):
            t.set_description(f"Loading features: {feature_file_name}")
            with np.load(feature_file_name) as features:
                all_features.append(features["arr_0"])

        all_features = np.concatenate(all_features, axis=0)

        # load datasets
        dfs = []
        for _, filename in enumerate(tqdm(datasets_paths, desc="Loading csv datasets")):
            df = pd.read_csv(filename, index_col=0)
            dfs.append(df)
        dataset = pd.concat(dfs, ignore_index=True)

        # drop unwanted cameras
        names = dataset["frame_name"]
        cams = []
        for name in names:
            index = name.find("cam") + 4
            cams.append(int(name[index]))

        dataset["cam"] = pd.Series(cams)

        if self.cams:
            self.cams = [int(c) for c in self.cams]
            cams_to_drop_mask = ~dataset["cam"].isin(self.cams)
            dataset = dataset.loc[~cams_to_drop_mask, :]
            dataset.reset_index(drop=True, inplace=True)

            all_features = np.delete(all_features, cams_to_drop_mask.tolist(), axis=0)
            all_features = normalize(all_features, axis=1, norm="l1")

        # drop off air frames
        if self.drop_offair:
            offair_mask = dataset["ar_labels"] == "actor_repositioning"

            dataset = dataset.loc[~offair_mask, :]
            dataset.reset_index(drop=True, inplace=True)

            all_features = np.delete(all_features, offair_mask.tolist(), axis=0)

        return dataset, all_features


def load_and_split(
    train_actors: list, val_actors: list, train_cams: list, val_cams: list, split_ratio: float, drop_offair: bool, undersample: bool, micro_classes: bool
) -> tuple[np.ndarray, list, np.ndarray, list, list, list]:
    # Load dataset and features
    features_folder = "outputs/dataset/features"
    dataset_folder = "outputs/dataset/dataset"

    if val_actors:
        train_dataloader = DatasetLoader(dataset_folder, features_folder, train_actors, train_cams, drop_offair)
        val_dataloader = DatasetLoader(dataset_folder, features_folder, val_actors, val_cams, drop_offair)
        print("[STATUS] Load Train Set")
        train_dataset, train_features = train_dataloader.load()
        print("[STATUS] Load Val Set")
        val_dataset, val_features = val_dataloader.load()

        X_train = train_features
        X_val = val_features
        if micro_classes:
            y_train = train_dataset["micro_labels"].tolist()
            y_val = val_dataset["micro_labels"].tolist()
        else:
            y_train = train_dataset["macro_labels"].tolist()
            y_val = val_dataset["macro_labels"].tolist()

        cams_train = train_dataset["cam"].tolist()
        cams_val = val_dataset["cam"].tolist()

        if undersample:
            print("[STATUS] Undersampling train set")
            print(f"Initial Train set distribution: {Counter(y_train)}")
            us = NearMiss(version=1)
            X_train, y_train = us.fit_resample(X_train, y_train)
            print(f"Train set distribution after undersampling: {Counter(y_train)}")

        return normalize(X_train), y_train, normalize(X_val), y_val, cams_train, cams_val

    else:
        # do the train-validation split
        dataset_dataloader = DatasetLoader(dataset_folder, features_folder, train_actors, train_cams, drop_offair)
        print("[STATUS] Load Dataset")
        dataset, features = dataset_dataloader.load()
        split = int(dataset.shape[0] * split_ratio)
        print("[STATUS] Splitting in Train and Val sets")
        X_train = np.array(features[0:split, :])
        X_val = np.array(features[split:, :])

        if micro_classes:
            y_train = dataset["micro_labels"][0:split].tolist()
            y_val = dataset["micro_labels"][split:].tolist()
        else:
            y_train = dataset["macro_labels"][0:split].tolist()
            y_val = dataset["macro_labels"][split:].tolist()

        cams_train = dataset["cams"][0:split].tolist()
        cams_val = dataset["cams"][split:].tolist()

        if undersample:
            print("[STATUS] Undersampling train set")
            print(f"Initial Train set distribution: {Counter(y_train)}")
            us = NearMiss(version=1)
            X_train, y_train = us.fit_resample(X_train, y_train)
            print(f"Train set distribution after undersampling: {Counter(y_train)}")

        return normalize(X_train), y_train, normalize(X_val), y_val, cams_train, cams_val


def get_timeseries_labels_encoded(y_train, y_val, cfg) -> tuple[list, list, LabelEncoder, dict, list]:
    def to_series_labels(timestep_labels: list, n_batches: int, n_windows: int, seq_len: int, stride: int) -> list:
        series_labels = []
        s = 0
        for w in range(n_windows * n_batches):
            s = w * stride
            labels_seq = timestep_labels[s : s + seq_len]
            series_labels.append(mode(labels_seq))
        return series_labels

    n_train_batches = len(y_train) // cfg.batch_size
    n_val_batches = len(y_val) // cfg.batch_size
    n_windows = (cfg.batch_size - cfg.seq_len) // cfg.stride + 1

    y_train_series = to_series_labels(y_train, n_train_batches, n_windows, cfg.seq_len, cfg.stride)
    y_val_series = to_series_labels(y_val, n_val_batches, n_windows, cfg.seq_len, cfg.stride)

    print(f"\nBefore ENCODING -- len(y_train_series): {len(y_train_series)} len(y_val_series): {len(y_val_series)}")

    print(f"y_train_series value counts:\n {pd.Series(y_train_series).value_counts()}")
    # encoding
    enc = LabelEncoder()
    enc = enc.fit(y_train_series)
    mapping = dict(zip(enc.classes_, range(1, len(enc.classes_) + 1)))
    print(f"Classes mapping: {mapping}")

    # y_train_series_unique = np.unique(y_train_series)
    # y_val_series_unique = np.unique(y_val_series)

    # if y_train_series_unique.sort() != y_val_series_unique.sort():
    #     raise Exception("y_train_series_unique != y_val_series_unique")

    y_train_series_encoded = enc.fit_transform(y_train_series)
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train_series_encoded), y=y_train_series_encoded)
    d_class_weights = dict(enumerate(class_weights))
    print(f"\nClass weights for train series: {class_weights}")

    y_train_series = enc.fit_transform(y_train_series)
    y_val_series = enc.fit_transform(y_val_series)

    print(f"\nAfter ENCODING -- len(y_train_series): {len(y_train_series)} len(y_val_series): {len(y_val_series)}")
    return y_train_series, y_val_series, enc, d_class_weights, enc.classes_.tolist()
