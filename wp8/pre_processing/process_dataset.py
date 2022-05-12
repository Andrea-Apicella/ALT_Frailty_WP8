import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm, trange
from wp8.pre_processing.utils import listdir_nohidden_sorted


class ProcessDataset:
    def __init__(self, videos_folder, feature_extractor, preprocess_input):
        self.videos_folder = videos_folder
        self.videos_paths = listdir_nohidden_sorted(self.videos_folder)
        self.feature_extractor = feature_extractor
        self.preprocess_input = preprocess_input

    def predict_frame(self, frame):
        size = self.feature_extractor.input_shape[1:3]
        frame = frame = tf.keras.preprocessing.image.smart_resize(
            frame, size)
        frame = self.preprocess_input(frame)
        return self.feature_extractor.predict(np.expand_dims(frame, axis=0))

    def extract_frames(self):

        dfs = []
        all_features = []
        for _, folder in enumerate((t0 := tqdm(self.videos_paths, position=0))):
            folder_name = folder.replace(self.videos_folder, "")[1:]
            t0.set_description(
                f'Processing folder: {folder_name}')

            sheet_name = folder.replace(self.videos_folder, "").lower()[1:]
            try:
                labels_sheet = pd.read_excel("outputs/labels/labels.xlsx",
                                             sheet_name=sheet_name, index_col=0)
                os.path.exists(
                    f"../outputs/dataset/dataset/{folder_name}.csv")

            except OSError:
                print(
                    f'Labels sheet {sheet_name} not found. Skippig {folder_name}.')
                continue

            if os.path.exists(f"outputs/dataset/dataset/{folder_name}.csv"):
                print(
                    f'Folder already processed. Skippig {folder_name}')
                continue

            video_iso_files_path = f'{folder}/Video ISO Files'

            frames_names = []
            folder_features = []

            video_iso_files = listdir_nohidden_sorted(
                video_iso_files_path)[:-1]

            for _, cam in enumerate((t1 := tqdm(video_iso_files, position=1, leave=True))):
                start = cam.rfind('/') + 1
                end = len(cam) - 4
                t1.set_description(
                    f'Extracting frames from: {cam[start:].replace(" ", "_")}')
                cap = cv2.VideoCapture(cam)
                try:
                    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    for f in trange(n_frames):
                        ret, frame = cap.read()
                        if not ret:
                            break

                        features = self.predict_frame(frame)
                        folder_features.append(features)

                        file_name = f'{cam[start:end].lower().replace(" ", "_")}_{str(f).zfill(4)}'
                        frames_names.append(file_name)

                finally:
                    cap.release()

            df = pd.concat(
                [labels_sheet] * len(video_iso_files), ignore_index=True)  # type: ignore

            df["frame_name"] = pd.Series(frames_names)

            # savez_compressed features as npy files
            folder_features = np.asarray(folder_features).squeeze()
            np.savez_compressed(
                f"outputs/dataset/features/{folder_name}.npy", folder_features)

            print(f"folder_features shape: {folder_features.shape}")

            # savez_compressed dataset as csv
            df.to_csv(f"outputs/dataset/dataset/{folder_name}.csv")

            all_features.append(folder_features)
            dfs.append(df)

        # savez_compressed all features as npy
        all_features = np.asarray(all_features).squeeze()
        np.savez_compressed(
            "outputs/dataset/features/all_features.npy", all_features)

        # savez_compressed full dataset as csv
        dataset = pd.concat(dfs)
        dataset.to_csv("outputs/dataset/dataset/full_dataset.csv")
