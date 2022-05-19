from utils.dataset_loader import DatasetLoader

features_path = "../outputs/dataset/features/"
dataset_path = "../outputs/dataset/dataset/"
train_loader = DatasetLoader(dataset_folder=dataset_path, features_folder=features_path, actors=[1], cams=[6, 7])

train_dataset, train_features = train_loader.load()
