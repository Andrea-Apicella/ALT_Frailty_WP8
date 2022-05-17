import traceback
from collections import OrderedDict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils.object_detection import non_max_suppression
from tqdm import trange

from pre_processing.utils import listdir_nohidden_sorted


class TemplateMatch:
    """Extraction of timestamps and datalogger time vectors from the NGIMU software video capture using Template Matching algorithm"""

    def __init__(
        self,
        video_path,
        element_type,
        timestamp_roi=None,
        datalogger_roi=None,
        target_template_size=(174, 255),
        target_timestamp_size=(2444, 428),
    ):
        self.templates_path = "./data/templates"

        self.templates = [cv2.imread(template) for template in listdir_nohidden_sorted(self.templates_path)]
        print(f"Found {len(self.templates)} templates in {self.templates_path}.")

        self.TARGET_TEMPLATE_SIZE = target_template_size
        self.TARGET_TIMESTAMP_SIZE = target_timestamp_size
        self.TARGET_DATALOGGER_SIZE = None
        self.video_path = video_path
        self.timestamp_roi = timestamp_roi
        self.datalogger_roi = datalogger_roi

        if element_type in ["timestamp", "datalogger"]:
            self.element_type = element_type
        else:
            print('Specified type of element to extract is invalid. Choose "timestamp" or "datalogger".')
            return

        def total_frames(video_path):
            cap = cv2.VideoCapture(video_path)
            _, _ = cap.read()
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return n_frames

        self.total_frames = total_frames(self.video_path)

    def __pre_process_templates(self) -> None:
        def pre_process(template):
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            template = cv2.resize(template, self.TARGET_TEMPLATE_SIZE)
            template = cv2.bitwise_not(template)
            return template.astype(np.uint8)

        self.templates = list(map(lambda template: pre_process(template), self.templates))

    def __select_roi(self, target):
        n_frames = self.total_frames
        print("Total number of frames in the video:", n_frames)
        cap = cv2.VideoCapture(self.video_path)
        _, first_frame = cap.read()
        roi = cv2.selectROI(f"{target}", first_frame)
        cap.release()
        return roi

    def template_match(self, frame, threshold=0.7, test=False):
        def pre_process(frame):
            frame = cv2.bitwise_not(frame)
            newsize = self.TARGET_TIMESTAMP_SIZE if self.element_type == "timestamp" else self.TARGET_DATALOGGER_SIZE
            frame = cv2.resize(frame, dsize=newsize, interpolation=cv2.INTER_CUBIC)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame.astype(np.uint8)

        def show(frame, tomatch, templates):
            """Shows templates, frame and frame ROI from which to extract the digits."""
            _, ax = plt.subplots(nrows=1, ncols=len(templates), figsize=(20, 30))
            for count, col in enumerate(ax):  # type: ignore
                col.imshow(self.templates[count], cmap="gray")
                col.set_title(f"template {count}")
            plt.title("Templates")
            plt.show()

            plt.figure(figsize=(20, 20))
            plt.title("Frame")
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.show()

            plt.figure()
            plt.title("ROI to perform Template Matching on")
            plt.imshow(tomatch, cmap="gray")
            plt.show()

        if len(self.templates[0].shape) > 2:
            self.__pre_process_templates()
        top_x, top_y, bottom_x, bottom_y = self.timestamp_roi if self.element_type == "timestamp" else self.datalogger_roi
        tomatch = frame[top_y : top_y + bottom_y, top_x : top_x + bottom_x]
        tomatch = pre_process(tomatch)

        if test:
            show(frame, tomatch, self.templates)

        tomatch_copy = tomatch.copy()

        # dict will contain starting x coordinates of template's digits bounding boxes
        x1Coords = {}

        for template_number, _ in enumerate(self.templates):
            curr_template = self.templates[template_number]
            (tH, tW) = curr_template.shape[:2]
            result = cv2.matchTemplate(tomatch_copy, curr_template, cv2.TM_CCOEFF_NORMED)
            (yCoords, xCoords) = np.where(result >= threshold)

            boxes = []
            for (x, y) in zip(xCoords, yCoords):
                boxes.append((x, y, x + tW, y + tH))
            nms_boxes = non_max_suppression(np.array(boxes))

            if len(nms_boxes) > 1:
                keys = [box[0] for box in nms_boxes]
                for key in keys:
                    x1Coords[key] = template_number
            elif len(nms_boxes) == 1:
                key = nms_boxes[0][0]
                x1Coords[key] = template_number
            else:
                pass

            if test:
                for (startX, startY, endX, endY) in nms_boxes:
                    cv2.rectangle(tomatch_copy, (startX, startY), (endX, endY), (0, 0, 0), 3)
            if template_number == len(self.templates) - 1:
                plt.figure()
                plt.imshow(tomatch_copy, cmap="gray")
                plt.title("Bounding boxes")
                plt.show()

        x1Coords_unique = OrderedDict(sorted(x1Coords.items()))
        digits = list(x1Coords_unique.values())
        try:
            extracted_format = list("XX:XX:XX.XXX")
            digits_index = 0
            extracted = extracted_format
            for _, char in enumerate(extracted_format):
                if char not in ([":", "."]):
                    extracted[_] = str(digits[digits_index])
                    digits_index += 1
            extracted = "".join(extracted)
            return extracted

        except Exception:
            traceback.print_exc()
            print(f"[ERROR] starts is empty! Zero bounding boxes drawn for this {self.element_type}.")

    def extract_timestamps(self):
        if not self.timestamp_roi:
            self.timestamp_roi = self.__select_roi("Select Timestamp ROI")
        print(f"Timestamp ROI: {self.timestamp_roi}")

        timestamp_test = self.__test_single_timestamp()
        if not timestamp_test:
            print("Can't extract timestamp from first frame.")
            return
        timestamps = []
        cap = cv2.VideoCapture(self.video_path)
        for frame_number in (t := trange(self.total_frames)):
            # for frame_number in trange(self.total_frames):
            success, frame = cap.read()
            if not success:
                print(f"Couldn't read frame {frame_number}")
                break
            timestamp_extracted = self.template_match(frame)
            timestamps.append(timestamp_extracted)
            t.set_description(f"Extracted timestamp: {timestamp_extracted}")
        cap.release()
        return timestamps

    def __test_single_timestamp(self):
        timestamp_extracted = None
        cap = cv2.VideoCapture(self.video_path)
        _, first_frame = cap.read()
        cap.release()

        timestamp_extracted = self.template_match(first_frame)
        return timestamp_extracted


def main():
    pass


if __name__ == "__main__":
    main()
