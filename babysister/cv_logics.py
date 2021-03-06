"""CV logics"""


def is_inside_roi(roi_value, box):
    """Is bouding box centroid inside ROI.

    Args:
        roi_value ([x, y, w, h] list): x, y: top-left; w, h: width & heigh.
        box ([x0, y0, x1, y1] list): x0, y0: top-left; x1, y1: bottom-right.

    Returns:
        bool: Whether box is inside ROI.
    """
    x, y, w, h = roi_value
    x0, y0, x1, y1 = box
    return x <= (x1 + x0) / 2 <= x + w and y <= (y1 + y0) / 2 <= y + h


def iou(bbox1, bbox2):
    """Calculates the intersection-over-union of two bounding boxes.

    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.

    Returns:
        int: intersection-over-onion of bbox1, bbox2.
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap
    # to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union
