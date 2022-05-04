import cv2
import pandas as pd
from tqdm import tqdm, trange
from wp8.pre_processing.utils import listdir_nohidden_sorted, safe_mkdir


class PrepareDataset():
    def __init__(self, videos_folder, frames_folder):
        self.videos_folder = videos_folder
        self.frames_folder = frames_folder
        self.videos_paths = listdir_nohidden_sorted(self.videos_folder)
        print(self.videos_paths[0])

    def extract_frames(self):
        for _, folder in enumerate((t0 := tqdm(self.videos_paths[:1], position=0))):
            t0.set_description(f'Processing folder {folder}')
            video_iso_files = f'{folder}/Video ISO Files'
            pre_path = '/Volumes/SSD 1TB 1/Alt Frailty WP8/video_dataset_initial_tests/'
            names = []  # list containing frames names for a single video
            for _, cam in enumerate((t1 := tqdm(listdir_nohidden_sorted(video_iso_files)[:1], position=1, leave=False))):
                t1.set_description(f'Extracting frames from {cam}')
                cap = cv2.VideoCapture(cam)
                try:
                    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    for i in trange(10):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        start = cam.rfind('/')+1
                        end = len(cam) - 4
                        file_name = cam[start:end].lower().replace(" ", "_")
                        names.append(file_name)
                        # cv2.imwrite(file_name, frame)
                finally:
                    cap.release()
            names = pd.Series(names)

            df = pd.read_excel(

                "wp8/excel_sheets/labels/labels.xlsx", sheet_name=f'{folder.replace(pre_path, "").lower()}', index_col=0)

            augmented = pd.concat(
                [df] * (len(video_iso_files) - 1), ignore_index=True)

            augmented["frame_name"] = names

            print(augmented.head())

            safe_mkdir(
                "/Users/andrea/Documents/Github/WP8_refactoring/wp8/excel_sheets/dataset/")
            try:
                with pd.ExcelWriter(path="/Users/andrea/Documents/Github/WP8_refactoring/wp8/excel_sheets/dataset/dataset.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                    augmented.to_excel(
                        writer, sheet_name='actor_1_bed_full_ph', index=True)
            except Exception as e:
                print(e)
