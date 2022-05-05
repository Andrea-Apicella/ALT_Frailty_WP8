from wp8.pre_processing.prepare_dataset import PrepareDataset

videos_folder = '/Volumes/HDD ESTERNO Andrea/DATASET WP8'
frames_folder = '/Volumes/SSD 1TB 1/Alt Frailty WP8/wp8_dataset_frames'

ds = PrepareDataset(videos_folder=videos_folder, frames_folder=frames_folder)
ds.extract_frames()
