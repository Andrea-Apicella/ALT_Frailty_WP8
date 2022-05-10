import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm, trange
from wp8.pre_processing.utils import listdir_nohidden_sorted, safe_mkdir


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

        safe_mkdir("outputs/dataset")
        dfs = []
        for _, folder in enumerate((t0 := tqdm(self.videos_paths[0:2], position=0))):
            folder_name = folder.replace(self.videos_folder, "")[1:]
            t0.set_description(
                f'Processing folder: {folder_name}')

            sheet_name = folder.replace(self.videos_folder, "").lower()[1:]
            try:
                labels_sheet = pd.read_excel("outputs/labels/labels.xlsx",
                                             sheet_name=sheet_name, index_col=0)
                not os.path.exists(f"outputs/dataset/{folder_name}.pkl")

            except Exception as e:
                print(e)
                print(
                    f'Labels sheet {sheet_name} not found or folder already processed. Skippig {folder_name}.')
                continue

            video_iso_files_path = f'{folder}/Video ISO Files'

            frames_names = []
            features_list = []

            video_iso_files = listdir_nohidden_sorted(
                video_iso_files_path)[:-1]

            for _, cam in enumerate((t1 := tqdm(video_iso_files[0:2], position=1, leave=True))):
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
                        features_list.append(features)
                        file_name = f'{cam[start:end].lower().replace(" ", "_")}_{str(f).zfill(4)}'
                        frames_names.append(file_name)

                finally:
                    cap.release()

            df = pd.concat(
                [labels_sheet] * len(video_iso_files[0:2]), ignore_index=True)  # type: ignore

            df["frame_name"] = pd.Series(frames_names)
            df["features"] = pd.Series(features_list)

            # if len(frames_names) != len(labels_sheet.index):
            #     print(
            #         f'frames_names length: {len(frames_names)}, labels_sheet length: {len(labels_sheet.index)}')
            #     raise Exception(
            #         "frame frames_names and labels don't have the same length")
            # else:
            #     print(
            #         f'frames_names length: {len(frames_names)}, labels_sheet length: {len(labels_sheet.index)}')
            df.to_json(f"outputs/dataset/{folder_name}.json")
            dfs.append(df)
        dataset = pd.concat(dfs)
        dataset.to_json("outputs/dataset/full_dataset.json")

        # try:
        #     with pd.ExcelWriter(
        #             path="outputs/dataset/dataset.xlsx",
        #             engine="openpyxl",
        #             mode="a",
        #             if_sheet_exists="replace") as writer:  # pylint: disable=abstract-class-instantiated
        #         df.to_excel(
        #             writer, sheet_name=f'{folder.replace(self.videos_folder, "").lower()[1:]}', index=True)
        # except Exception as e:
        #     print(e)
