"""Wrapper for sort"""
import numpy as np
from pprint import pprint

from .sort.sort import Sort


class SORT:
    def __init__(self):
        self.tracker = Sort()

    def _gen_detections(self, boxes, scores):
        """
        Args:

        Return: numpy array
            detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        """
        detections = [None] * len(boxes)
        for i, (box, score) in enumerate(zip(boxes, scores)):
            detections[i] = [*box, score]
        return np.asarray(detections)

    def update(self, boxes, scores):
        """
        """
        detections = self._gen_detections(boxes, scores)
        return self.tracker.update(detections)
