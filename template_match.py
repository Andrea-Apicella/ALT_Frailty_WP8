from ctypes.wintypes import SIZE
from time import time
from PIL import Image, ImageChops
from utils.utils import *
import matplotlib.pyplot as plt
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from collections import OrderedDict
from tqdm import trange


class TemplateMatch():

    def __init__(self, video_path, timestamp_roi=None, element_type=None, datalogger_roi=None, TARGET_TEMPLATE_SIZE=(174, 255), TARGET_TIMESTAMP_SIZE=(2444, 428)):
        self.templates_path = './data/templates'
        try:
            self.templates = [cv2.imread(template, 0)
                              for template in listdir_nohidden_sorted(self.templates_path)]
        except:
            print('Can\'t read templates.')

        self.TARGET_TEMPLATE_SIZE = TARGET_TEMPLATE_SIZE
        self.TARGET_TIMESTAMP_SIZE = TARGET_TIMESTAMP_SIZE
        self.video_path = video_path
        try:
            element_type in ['timestamp', 'datalogger']
        except:
            print('Specified type of element to extract is not valid.')
        else:
            self.element_type = element_type

        def total_frames(video_path):
            cap = cv2.VideoCapture(video_path)
            _, first_frame = cap.read()
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return total_frames

        self.total_frames = total_frames(self.video_path)

    def __pre_process_templates(self):
        def pre_process(template):
            template = cv2.resize(template, self.TARGET_TEMPLATE_SIZE)
            template = cv2.bitwise_not(template)
            return template

        self.templates = list(map(
            lambda template: pre_process(template), self.templates))
        print(self.templates)
        return self.templates

    def __select_roi(self, target):
        n_frames = self.total_frames
        print('Total number of frames in the video:', n_frames)
        cap = cv2.VideoCapture(self.video_path)
        _, first_frame = cap.read()
        roi = cv2.selectROI(f'{target}', first_frame)
        cv2.waitKey(0)
        cv2.destroyWindow(f'{target}')
        cap.release()
        return roi

    def __template_match(self, frame, threshold=0.7):
        def pre_process(frame_roi):
            cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
            cv2.bitwise_not(frame_roi)
            cv2.resize(frame_roi, size=self.TARGET_TEMPLATE_SIZE if self.element_type == 'timestamp' else self.TARGET_DATALOGGER_SIZE,
                       interpolation=cv2.INTER_CUBIC)
            return frame_roi
        top_x, top_y, bottom_x, bottom_y = self.timestamp_roi
        timestamp = frame[top_y:top_y+bottom_y, top_x:top_x+bottom_x]
        timestamp = pre_process(timestamp)

        x1Coords = {}  # dict will contain starting x coordinates of template's digits bounding boxes

        for template_number in range(self.templates):
            curr_template = self.templates[template_number]
            (tH, tW) = curr_template.shape[:2]
            result = cv2.matchTemplate(
                timestamp, curr_template, cv2.TM_CCOEFF_NORMED)
            (yCoords, xCoords) = np.where(result >= threshold)

            boxes = [boxes.append(x, y, x + tW, y + tH)
                     for (x, y) in zip(xCoords, yCoords)]
            # for (x, y) in zip(xCoords, yCoords):
            #     boxes.append(x, y, x + tW, y + tH)
            nms_boxes = non_max_suppression(np.array(boxes))

            if len(nms_boxes) > 1:
                keys = [box[0] for box in nms_boxes]
                for key in keys:
                    x1Coords[key] = template_number
            elif len(nms_boxes) == 1:
                key = nms_boxes[0][0]
                x1Coords[key] = template_number
        x1Coords_unique = OrderedDict(sorted(x1Coords.items()))
        digits = list(x1Coords_unique.values())
        try:
            format = list('XX:XX:XX.XXX')
            digits_index = 0
            exctracted = format
            for _, char in enumerate(format):
                if not char in ([':', '.']):
                    exctracted[_] = str(digits[digits_index])
                    digits_index += 1
            extracted = ''.join(extracted)
            return extracted
        except:
            print(
                f'[ERROR] starts is empty! Zero bounding boxes drawn for this {self.element_type}.')

    def extract_timestamps(self):
        if not self.timestamp_roi:
            self.timestamp_roi = self.__select_roi('Select Timestamp ROI')
        print(f'Timestamp ROI: {self.timestamp_roi}')

        self.__pre_process_templates()
        timestamps = []
        cap = cv2.VideoCapture(self.video_path)
        for frame_number in (t := trange(self.total_frames)):
            success, frame = cap.read()
            if not success:
                print(f'Couldn\'t read frame {frame_number}')
                break
            timestamp_extracted = self.__template_match(
                frame)
            timestamps.append(timestamp_extracted)
            t.set_description(f'Extracted timestamp: {timestamp_extracted}')
        return timestamps


if __name__ == "__main__":
    tm = TemplateMatch(
        video_path='/Users/andrea/Desktop/Actor_1_Bed_Full_PH.mp4')
    templates = tm.load_templates()
    # for _, template in enumerate(templates):
    #     template.save(f'./template{_}.jpg')
    fig, ax = plt.subplots(nrows=1, ncols=10, figsize=(30, 10))
    for count, col in enumerate(ax):
        col.imshow(templates[count], cmap='gray')
        col.set_title(f'Template {str(col)}')

    # plt.show()

    roi = tm.extract_timestamps()
