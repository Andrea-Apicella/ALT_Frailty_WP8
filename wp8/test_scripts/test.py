import pickle as pkl

import h5py
import numpy as np
import pandas as pd

features = np.zeros((10, 2048))

# hf = h5py.File("./features.h5", "w")

# hf.create_dataset("features_inceptionV3", data=features)
# hf.close()

hf_r = h5py.File("features.h5", "r")

features_r = hf_r.get("features_inceptionV3")

print(features.shape, np.array(features_r.shape))
print(np.array(features_r).shape)

print()
