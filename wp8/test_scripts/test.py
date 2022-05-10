from xml.sax.handler import all_features

import numpy as np

all_features = np.load(
    "outputs/dataset/features/Actor_1_Bed_Full_PH.npy")
print(f"all_features shape: {all_features.shape}")


print(all_features.squeeze().shape)
