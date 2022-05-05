import os

import pandas as pd
from wp8.pre_processing.utils import listdir_nohidden_sorted

# df = pd.read_excel("", sheet_name="labels")
# print(df.head())


# df = pd.read_excel("wp8/excel_sheets/labels/labels.xlsx",
#                    sheet_name="actor_1_bed_full_ph")

# print()


print(listdir_nohidden_sorted(
    '/Volumes/SSD 1TB 1/Alt Frailty WP8/video_dataset_initial_tests/Actor_1_Bed_Full_PH/Video ISO Files'))

df = pd.read_excel("excel_sheets/labels/labels.xlsx",
                   sheet_name="actor_1_bed_full_ph")

print(len(df.index))
