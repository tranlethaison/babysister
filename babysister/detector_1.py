import cv2 as cv
import numpy as np


class Detector:
    """Additional logics for Object Detection.

    Args:
        detector_core: Core detector.
    """

    def __init__(self, detector_core):
        self.detector_core = detector_core
        self.input_size = [detector_core.size] * 2

    def detect(self, img_np, valid_classes, max_bb_size_ratio):
        """Wrapper for :func:`self.detector_core.detect` with additional logics.

        Additional logics:
            Only detect classes in `valid_classes`, omit others.
            Fiter boxes with size ratio greater than `max_bb_size_ratio`.

        Args:
            img_np (ndarray): input numpy images (BGR).
            valid_classes (list of str): Only detect these classes.
            max_bb_size_ratio (list of 2 int):
                Maximum box width ratio, height ratio wrt `img_np` size.

        Returns:
            [boxes, scores, labels]:
                `boxes` (ndarray) is boxes coordinate in format [[x0, y0, x1, y1], ...].
                `scores` (ndarray) is confidence scores.
                `labels` (ndarray) is label indexes.
        """
        img_input = cv.cvtColor(img_np, cv.COLOR_BGR2RGB)

        boxes, scores, labels = self.detector_core.detect(img_input)

        # filter by class
        if valid_classes is not None:
            tmp_boxes, tmp_scores, tmp_labels = [], [], []

            for box, score, label in zip(boxes, scores, labels):
                if self.detector_core.class_names[int(label)] in valid_classes:
                    tmp_boxes.append(box)
                    tmp_scores.append(score)
                    tmp_labels.append(label)

            boxes, scores, labels = tmp_boxes, tmp_scores, tmp_labels

        # rescale boxes
        boxes = np.array(boxes)
        if boxes.shape[0] > 0:
            h, w = img_np.shape[:2]
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
