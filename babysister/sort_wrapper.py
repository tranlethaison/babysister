import numpy as np
from pprint import pprint

from .sort.sort import Sort


class SORT:
    """Wrapper for :class:`sort.Sort`."""

    def __init__(self):
        self.tracker = Sort()

    def _gen_detections(self, boxes, scores):
        """Genrate detections data.

        Args:
            boxes (list): Boxes coordinate in format [[x0, y0, x1, y1], ...].
            scores (list): Confidence scores.

        Return:
            ndarray: detections in the format [[x0, y0, x1, y1, score], ...]
        """
        detections = [None] * len(boxes)
        for i, (box, score) in enumerate(zip(boxes, scores)):
            detections[i] = [*box, score]
        return np.asarray(detections)

    def update(self, boxes, scores):
        """Update tracked objects.

        Args:
            boxes (list): Boxes coordinate in format [[x0, y0, x1, y1], ...].
            scores (list): Confidence scores.

        Return:
            ndarray: detections in the format [[x0, y0, x1, y1, id], ...]

        Requires: 
            This method must be called once for each frame even with empty detections.

        Note:
            The number of objects returned may differ from the number of boxes provided.
        """
        detections = self._gen_detections(boxes, scores)
        return self.tracker.update(detections)
