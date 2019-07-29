import cv2 as cv
import numpy as np


class Detector:
    """Additional logics for OBJ Detection.

    Args:
        core_detector (:class:`yolov3_wrapper.YOLOv3` instance): Core detector.
    """

    def __init__(self, core_detector):
        self.core_detector = core_detector
        self.input_size = core_detector.input_size[::-1]

    def detect(self, im, valid_classes, max_bb_size_ratio):
        """Wrapper for :func:`self.core_detector.detect` with additional logics.

        Additional logics:
            Only detect classes in `valid_classes`, omit others.
            Fiter boxes with size ratio greater than `max_bb_size_ratio`.

        Args:
            im (ndarray): input images.
            valid_classes (list of str): Only detect these classes.
            max_bb_size_ratio (list of 2 int): 
                Maximum box width ratio, height ratio wrt `im` size.

        Returns:
            [boxes, scores, labels]:
                `boxes` (ndarray) is boxes coordinate in format [[x0, y0, x1, y1], ...].
                `scores` (ndarray) is confidence scores.
                `labels` (ndarray) is label indexes.
        """
        # input data
        input_data = cv.resize(
            im, dsize=tuple(self.input_size), interpolation=cv.INTER_LANCZOS4
        )
        input_data = cv.cvtColor(input_data, cv.COLOR_BGR2RGB)
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

        # detect
        boxes, scores, labels = self.core_detector.detect(input_data)

        # filter by class
        if valid_classes is not None:
            tmp_boxes, tmp_scores, tmp_labels = [], [], []

            for box, score, label in zip(boxes, scores, labels):
                if self.core_detector.classes[label] in valid_classes:
                    tmp_boxes.append(box)
                    tmp_scores.append(score)
                    tmp_labels.append(label)

            boxes, scores, labels = tmp_boxes, tmp_scores, tmp_labels

        # rescale boxes
        boxes = np.array(boxes)
        if boxes.shape[0] > 0:
            h, w = im.shape[:2]
            size_ratio = np.divide([w, h], self.input_size)
            boxes[:, 0] *= size_ratio[0]
            boxes[:, 1] *= size_ratio[1]
            boxes[:, 2] *= size_ratio[0]
            boxes[:, 3] *= size_ratio[1]

        # filter by box size wrt image size.
        if np.greater([1, 1], max_bb_size_ratio).any():
            tmp_boxes, tmp_scores, tmp_labels = [], [], []
            h, w = im.shape[:2]

            for box, score, label in zip(boxes, scores, labels):
                x0, y0, x1, y1 = box
                size_ratio = np.divide([x1 - x0, y1 - y0], [w, h])
                if np.greater(size_ratio, max_bb_size_ratio).any():
                    continue

                tmp_boxes.append(box)
                tmp_scores.append(score)
                tmp_labels.append(label)

            boxes, scores, labels = tmp_boxes, tmp_scores, tmp_labels

        boxes, scores, labels = list(map(np.array, [boxes, scores, labels]))

        return boxes, scores, labels
