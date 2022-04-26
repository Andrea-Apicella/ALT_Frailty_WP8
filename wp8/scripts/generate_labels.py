import json

import pandas as pd
from tqdm import tqdm
from wp8.scripts.utils import listdir_nohidden_sorted


class LabelsGenerator:
    """Generate labels from JSON files containing class ranges manually annotated in Supervisely"""

    def __init__(self, json_dir):
        self.json_dir = json_dir
        self.json_files_paths = listdir_nohidden_sorted(self.json_dir)
        # self.json_files_paths = [
        #     file for file in self.json_files_paths if not file.endswith(".json")
        # ]
        print(f"Found {len(self.json_files_paths)} JSON files")

    def gen_labels_single_file(self, file):

        with open(file) as json_file:
            self.labels_dict = json.load(json_file)

        tags = self.labels_dict["tags"]

        def no_ac(tag):
            return tag["name"] != "actor_repositioning"

        tags_no_ac = list(filter(no_ac, tags))

        for i, tag in enumerate(tags_no_ac):
            curr_end = tags_no_ac[i]["frameRange"][1]
            if i < len(tags_no_ac) - 1:
                next_start = tags_no_ac[i + 1]["frameRange"][0]
                if next_start == curr_end:
                    print(next_start)
                    tags_no_ac[i + 1]["frameRange"][0] += 1
        print(tags_no_ac)

        def interval_to_list(interval, label):
            start = interval[0]
            end = interval[1]
            length = end - start + 1
            if label == "actor_repositioning":
                return []
            else:
                return [str(label)] * length

        # sheet = pd.DataFrame()

        labels_list = []
        for tag in tags_no_ac:
            frame_range = tag["frameRange"]
            label = tag["name"]
            labels_list += interval_to_list(frame_range, label)
            if label == "actor_repositioning":
                pass
            else:
                pass

        print(
            f'Labels generate: {len(labels_list)}. FramesCount: {self.labels_dict["framesCount"]}\nIl numero di etichette generate è uguale al numero di frames: {len(labels_list) == self.labels_dict["framesCount"]}'
        )


def test():
    lg = LabelsGenerator(json_dir="wp8/data/labels_json/")
    print(lg.json_files_paths)
    lg.gen_labels_single_file(
        file="wp8/data/labels_json/WP8_labeling_Actor_1_Bed_Full_PH.mp4.json"
    )


if __name__ == "__main__":
    test()
