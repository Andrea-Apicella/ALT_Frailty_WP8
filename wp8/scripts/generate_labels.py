import json

import pandas as pd
from tqdm import tqdm
from wp8.scripts.utils import listdir_nohidden_sorted


class LabelsGenerator:
    """Generate labels from JSON files containing class ranges manually annotated in Supervisely"""

    def __init__(self, json_dir):
        self.json_dir = json_dir
        self.json_files_paths = listdir_nohidden_sorted(self.json_dir)
        self.labels_dict = None

        print(f"[INFO] Found {len(self.json_files_paths)} JSON files")

    def gen_labels_single_file(self, file):
        """Generate labels from a single JSON file containing tags from Supervisely"""

        def interval_to_list(interval, label):
            """Generate list of labels from an interval of frames"""
            start = interval[0]
            end = interval[1]
            length = end - start + 1
            return [str(label)] * length

        def check_adjacency(tags):
            non_consecutive = []
            # check that frameRanges are subsequent
            for i, tag in enumerate(tags[:-1]):
                curr_end = tags[i]["frameRange"][1]
                next_start = tags[i+1]["frameRange"][0]
                if next_start - curr_end != 1:
                    non_consecutive.append(
                        [tags[i]["frameRange"], tags[i+1]["frameRange"]])

            if len(non_consecutive) > 0:
                raise Exception(
                    f'[ERROR] In file {self.labels_dict["videoName"]} the following frameRanges are not consecutive:\n{non_consecutive}')

        with open(file) as json_file:
            self.labels_dict = json.load(json_file)

        tags_with_duplicates = self.labels_dict["tags"]
        tags = tags_with_duplicates.copy()

        # check for duplicate tags
        # for i, _ in enumerate(tags):
        #     curr_frame_range = tags[i]["frameRange"]
        #     for j, _ in enumerate(tags):
        #         other_frame_range = tags[j]["frameRange"]
        #         if i != j and curr_frame_range == other_frame_range and tags[i]["name"] == tags[j]["name"]:
        #             tags.pop(j)

        # filter out objects where label is "actor_repositioning"
        tags_no_ar = list(
            filter(lambda tag: tag["name"] != "actor_repositioning", tags)
        )
        # filter out objects where label is NOT "actor_repositioning"
        tags_ar = list(
            filter(lambda tag: tag["name"] == "actor_repositioning", tags))

        # sort tags based on the frame ranges. Necessary as Supervisely messes up the order of the tags sometimes
        tags_no_ar.sort(key=lambda tag: tag["frameRange"][0])
        tags_ar.sort(key=lambda tag: tag["frameRange"][0])

        check_adjacency(tags_no_ar)
        # check_adjacency(tags_ar)

        # correct start frame number of next interval if it is equal to end frame number of previous interval
        for i, tag in enumerate(tags_no_ar):
            curr_end = tags_no_ar[i]["frameRange"][1]
            if i < len(tags_no_ar) - 1:
                next_start = tags_no_ar[i + 1]["frameRange"][0]
                if next_start == curr_end:
                    tags_no_ar[i + 1]["frameRange"][0] += 1

        micro_labels = []
        for tag in tags_no_ar:
            frames_range = tag["frameRange"]
            label = tag["name"]
            micro_labels += interval_to_list(frames_range, label)

        ar_labels = ["on_air"] * len(micro_labels)

        # replacing "on_air" label with "actor_repositioning" where is needed
        for tag in tags_ar:
            frames_range = tag["frameRange"]
            ar_labels[frames_range[0]: frames_range[1] + 1] = interval_to_list(
                frames_range, "actor_repositioning"
            )

        micro_labels = pd.Series(micro_labels)
        ar_labels = pd.Series(ar_labels)

        d = {"micro_labels": micro_labels, "ar_labels": ar_labels}
        labels = pd.DataFrame(data=d)

        sheet_name = self.labels_dict["videoName"].replace(".mp4", "").lower()

        with pd.ExcelWriter(
            "/Users/andrea/Documents/Github/WP8_refactoring/wp8/excel_sheets/labels/labels.xlsx",
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace",
        ) as writer:
            labels.to_excel(writer, sheet_name, index=True)

    def generate_labels(self):
        """Generate labels excel file from multiple JSON files in the specified folder"""
        for _, file in enumerate(tqdm(self.json_files_paths)):
            self.gen_labels_single_file(file)


def test():
    lg = LabelsGenerator(json_dir="wp8/data/labels_json/")
    print(lg.json_files_paths)
    lg.gen_labels_single_file(
        file="wp8/data/labels_json/WP8_labeling_Actor_1_Bed.mp4.json"
    )


if __name__ == "__main__":
    test()
