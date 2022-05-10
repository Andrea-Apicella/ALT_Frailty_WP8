
import tensorflow as tf
from wp8.pre_processing.process_dataset import ProcessDataset

videos_folder = '/Volumes/HDD ESTERNO Andrea/DATASET WP8'
# videos_folder = '/Volumes/SSD 1TB 1/Alt Frailty WP8/DATASET_WP8'

feature_extractor = tf.keras.applications.InceptionV3(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3),
)


ds = ProcessDataset(videos_folder=videos_folder, feature_extractor=feature_extractor,
                    preprocess_input=tf.keras.applications.inception_v3.preprocess_input)
ds.extract_frames()
