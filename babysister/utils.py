import time
import colorsys
import cv2 as cv


def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.

    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2

    Returns:
        int: intersection-over-onion of bbox1, bbox2
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


def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)


def putText_withBackGround(
    img, text, top_right, fontFace, fontScale, fontThickness, color
):
    """Deprecated, please use putTextWithBG instead
    Put text in white, with a 'color' background
    """
    text_size = cv.getTextSize(text, fontFace, fontScale, fontThickness)

    x1, y1 = top_right
    center = x1 + 5, y1 + 5 + text_size[0][1]
    x2, y2 = x1 + 10 + text_size[0][0], y1 + 10 + text_size[0][1]

    cv.rectangle(img, (x1,y1), (x2,y2), color, -1)
    cv.putText(
        img, text, center, fontFace, fontScale, (255,255,255), fontThickness)


def putTextWithBG(
    img, txt, top_left, fontFace, fontScale, fontThickness, color, colorBG
):
    '''Put text with background
    '''
    txt_size, _ = cv.getTextSize(txt, fontFace, fontScale, fontThickness)
    w, h = txt_size
    x, y = top_left
    img[y:y+h, x:x+w, :] = colorBG
    cv.putText(
        img, txt, (x, y+h), 
        fontFace, fontScale, color, fontThickness)
    return txt_size


class FPSCounter:
    def __init__(self, limit):
        self.limit = limit
        self.start()

    def start(self):
        self.start_time = time.time()
        self.counter = 0
        self.fps = 0

    def tick(self):
        self.counter += 1
        if (time.time() - self.start_time) > self.limit:
            self.fps = self.counter / (time.time() - self.start_time)
            self.counter = 0
            self.start_time = time.time()
        return self.fps

    def get(self):
        return self.fps
