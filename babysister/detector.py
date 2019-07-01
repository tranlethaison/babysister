"""Additional logics for object detections"""
import cv2 as cv
import numpy as np


class Detector:
    def __init__(self, core_detector):
        """"""
        self.core_detector = core_detector

        h, w = core_detector.input_size
        self.input_size = [w, h]

    def detect(self, im, valid_classes, max_bb_size_ratio):
        """"""
        # input data
        input_data = cv.resize(
            im, dsize=tuple(self.input_size), interpolation=cv.INTER_LANCZOS4)
        input_data = cv.cvtColor(input_data, cv.COLOR_BGR2RGB)
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

        # detect
        boxes, scores, labels = self.core_detector.detect(input_data)

        # filter by class
        if 'all' not in valid_classes:
            tmp_boxes, tmp_scores, tmp_labels = [], [], []
            for box, score, label in zip(boxes, scores, labels):
                if self.core_detector.classes[label] in valid_classes:
                    tmp_boxes.append(box)
                    tmp_scores.append(score)
                    tmp_labels.append(label)
            boxes, scores, labels = np.array(tmp_boxes), tmp_scores, tmp_labels

        # rescale boxes
        if boxes.shape[0] > 0:
            h, w = im.shape[:2]
            size_ratio = np.divide([w, h], self.input_size)
            boxes[:,0] *= size_ratio[0]
            boxes[:,1] *= size_ratio[1]
            boxes[:,2] *= size_ratio[0]
            boxes[:,3] *= size_ratio[1]

        #filter by box size wrt image size.
        if np.greater([1,1], max_bb_size_ratio).any():
            h, w = im.shape[:2]
            tmp_boxes, tmp_scores, tmp_labels = [], [], []
            for box, score, label in zip(boxes, scores, labels):
                x0, y0, x1, y1 = box
                size_ratio = np.divide([x1-x0, y1-y0], [w, h])
                if np.greater(size_ratio, max_bb_size_ratio).any():
                    continue

                tmp_boxes.append(box)
                tmp_scores.append(score)
                tmp_labels.append(label)
            boxes, scores, labels = np.array(tmp_boxes), tmp_scores, tmp_labels

        return boxes, scores, labels

