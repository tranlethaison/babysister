"""Wraper for SORT
@inproceedings{Bewley2016_sort,
  author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
  booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
  title={Simple online and realtime tracking},
  year={2016},
  pages={3464-3468},
  keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
  doi={10.1109/ICIP.2016.7533003}
}
"""
import numpy as np
from pprint import pprint

from .sort.sort import Sort


class SORTTracker:
    def __init__(self):
        self.tracker = Sort()

    def gen_detections(self, boxes, scores):
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
        detections = self.gen_detections(boxes, scores)
        return self.tracker.update(detections)
