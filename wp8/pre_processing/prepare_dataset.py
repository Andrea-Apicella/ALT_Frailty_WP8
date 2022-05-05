import os

import cv2
import pandas as pd
from tqdm import tqdm, trange
from wp8.pre_processing.utils import listdir_nohidden_sorted, safe_mkdir


class PrepareDataset:
    def __init__(self, videos_folder, frames_folder):
        self.videos_folder = videos_folder
        self.frames_folder = frames_folder
        self.videos_paths = listdir_nohidden_sorted(self.videos_folder)

    def extract_frames(self):
        safe_mkdir(self.frames_folder)
        for _, folder in enumerate((t0 := tqdm(self.videos_paths[0:53], position=0))):
            t0.set_description(
                f'Processing folder: {folder.replace(self.videos_folder, "")[1:]}')
            video_iso_files_path = f'{folder}/Video ISO Files'

            names = []  # list containing frames names for a single video
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

                        file_name = f'{cam[start:end].lower().replace(" ", "_")}_{str(f).zfill(4)}'
                        names.append(file_name)
                        frame_path = f'{self.frames_folder}/{file_name}.jpg'
                        if not os.path.exists(frame_path):
                            cv2.imwrite(frame_path, frame)
                finally:
                    cap.release()
            names_serie = pd.Series(names)

            labels_sheet = pd.read_excel("excel_sheets/labels/labels.xlsx",
                                         sheet_name=f'{folder.replace(self.videos_folder, "").lower()[1:]}', index_col=0)

            df = pd.concat(
                [labels_sheet] * len(video_iso_files), ignore_index=True)

            df["frame_name"] = names_serie

            # if len(names) != len(labels_sheet.index):
            #     print(
            #         f'names length: {len(names)}, labels_sheet length: {len(labels_sheet.index)}')
            #     raise Exception(
            #         "frame names and labels don't have the same length")
            # else:
            #     print(
            #         f'names length: {len(names)}, labels_sheet length: {len(labels_sheet.index)}')

            safe_mkdir("excel_sheets/dataset")
            try:
                with pd.ExcelWriter(
                        path="excel_sheets/dataset/dataset.xlsx",
                        engine="openpyxl",
                        mode="a",
                        if_sheet_exists="replace") as writer:  # pylint: disable=abstract-class-instantiated
                    df.to_excel(
                        writer, sheet_name=f'{folder.replace(self.videos_folder, "").lower()[1:]}', index=True)
            except Exception as e:
                print(e)
