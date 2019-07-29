"""Drawing routines"""
import colorsys

import numpy as np
import cv2 as cv


# Defaults
_bg_color = (0, 0, 0)
_txt_color = (255, 255, 255)
_fontFace = cv.FONT_HERSHEY_SIMPLEX
_fontScale = 0.35
_fontThickness = 1
_boxThickness = 2


def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Args:
        tag (int): The unique target identifying tag.
        hue_step (float):
            Difference between two neighboring color codes in HSV space 
            (more specifically, the distance in hue channel).

    Returns:
        (float, float, float): RGB color code in range [0, 1].
    """
    h, v = (tag * hue_step) % 1, 1.0 - (int(tag * hue_step) % 4) / 5.0
    r, g, b = colorsys.hsv_to_rgb(h, 1.0, v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Args:
        tag (int): The unique target identifying tag.
        hue_step (float):
            Difference between two neighboring color codes in HSV space 
            (more specifically, the distance in hue channel).

    Returns:
        (int, int, int): RGB color code in range [0, 255].
    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255 * r), int(255 * g), int(255 * b)


def put_line_bg(
    im,
    line,
    top_left,
    bg_color=_bg_color,
    txt_color=_txt_color,
    fontFace=_fontFace,
    fontScale=_fontScale,
    fontThickness=_fontThickness,
):
    """Put a line of text with background.

    Args:
        im (ndarray): image to draw to.
        line (str): a line of text.
        top_left (list of 2 int): top_left of text box.
        bg_color (int, list of 3 int): Background color RGB code.
        txt_color (int, list of 3 int): Text color RGB code.
        fontFace: OpenCV Font type.
        fontScale (float): 
            Font scale factor that is multiplied by the font-specific base size.
        fontThickness (int): Thickness of the lines used to draw text.

    Returns:
        ((w, h), baseLine):
            `(w, h)` is size of text box. 
            `baseLine` is y-coordinate of the baseline wrt the bottom-most text point. 
    """
    (w, h), baseLine = cv.getTextSize(line, fontFace, fontScale, fontThickness)
    x, y = top_left

    im[y : y + h + baseLine, x : x + w, :] = bg_color
    cv.putText(im, line, (x, y + h), fontFace, fontScale, txt_color, fontThickness)

    return (w, h), baseLine


def put_lines_bg(
    im,
    lines,
    top_left,
    bg_color=_bg_color,
    txt_color=_txt_color,
    fontFace=_fontFace,
    fontScale=_fontScale,
    fontThickness=_fontThickness,
):
    """Put lines of text with background.

    Args:
        im (ndarray): image to draw to.
        line (list of str): lines of text.
        top_left (list of 2 int): top_left of text box.
        bg_color (int, list of 3 int): Background color RGB code.
        txt_color (int, list of 3 int): Text color RGB code.
        fontFace: OpenCV Font type.
        fontScale (float): 
            Font scale factor that is multiplied by the font-specific base size.
        fontThickness (int): Thickness of the lines used to draw text.

    Returns:
        ((w, h), baseLine):
            `(w, h)` is size of bottom-most text box. 
            `baseLine` is y-coordinate of the baseline wrt the bottom-most text point. 
    """
    top_left = np.asarray(top_left, np.int32)
    for line in lines:
        (w, h), baseLine = put_line_bg(
            im, line, top_left, bg_color, txt_color, fontFace, fontScale, fontThickness
        )
        top_left += [0, h + baseLine]

    return (w, h), baseLine


def draw_detection(
    im,
    box,
    score,
    label,
    classes,
    do_show_class=True,
    box_color=None,
    txt_color=_txt_color,
    fontFace=_fontFace,
    fontScale=_fontScale,
    fontThickness=_fontThickness,
    boxThickness=_boxThickness,
):
    """Draw detections.

    Args:
        im (ndarray): image to draw to.
        box (list of int): box coordinate in format [x0, y0, x1, y1].
        score (float): confidence score.
        label (int): label index.
        classes (dict): labels data {id: name}.
        bg_color (int, list of 3 int): Background color RGB code.
        txt_color (int, list of 3 int): Text color RGB code.
        fontFace: OpenCV Font type.
        fontScale (float): 
            Font scale factor that is multiplied by the font-specific base size.
        fontThickness (int): Thickness of the lines used to draw text.
    """
    box_color = box_color or create_unique_color_uchar(label)

    x0, y0, x1, y1 = map(int, box)
    cv.rectangle(im, (x0, y0), (x1, y1), box_color, boxThickness)

    if do_show_class:
        line = "{:.02f} {}".format(score, classes[label])
        (txt_w, txt_h), baseLine = cv.getTextSize(
            line, fontFace, fontScale, fontThickness
        )
        top_left = [x0, y0 - txt_h - baseLine]

        put_line_bg(
            im, line, top_left, box_color, txt_color, fontFace, fontScale, fontThickness
        )


def draw_tracking(
    im,
    track,
    box_color=None,
    txt_color=_txt_color,
    fontFace=_fontFace,
    fontScale=_fontScale,
    fontThickness=_fontThickness,
    boxThickness=_boxThickness,
):
    """Draw tracking.

    Args:
        im (ndarray): image to draw to.
        track (ndarray): tracking data in format [[x0,y0,x1,y1,id], ...]
        bg_color (int, list of 3 int): Background color RGB code.
        txt_color (int, list of 3 int): Text color RGB code.
        fontFace: OpenCV Font type.
        fontScale (float): 
            Font scale factor that is multiplied by the font-specific base size.
        fontThickness (int): Thickness of the lines used to draw text.
    """
    id_ = int(track[4])
    box_color = box_color or create_unique_color_uchar(id_)

    x0, y0, x1, y1 = map(int, track[:4])
    cv.rectangle(im, (x0, y0), (x1, y1), box_color, boxThickness)

    put_line_bg(
        im, str(id_), (x0, y0), box_color, txt_color, fontFace, fontScale, fontThickness
    )


def draw_roi(
    im,
    roi,
    n_detected_objs,
    box_color=_bg_color,
    txt_color=_txt_color,
    fontFace=_fontFace,
    fontScale=_fontScale,
    fontThickness=_fontThickness,
    boxThickness=_boxThickness,
):
    """Draw ROI.

    Args:
        im (ndarray): image to draw to.
        roi (dict): ROI data with keys ["id", "x", "y", "w", "h"].
        bg_color (int, list of 3 int): Background color RGB code.
        txt_color (int, list of 3 int): Text color RGB code.
        fontFace: OpenCV Font type.
        fontScale (float): 
            Font scale factor that is multiplied by the font-specific base size.
        fontThickness (int): Thickness of the lines used to draw text.
    """
    x0, y0 = int(roi["x"]), int(roi["y"])
    x1, y1 = int(roi["x"] + roi["w"]), int(roi["y"] + roi["h"])
    cv.rectangle(im, (x0, y0), (x1, y1), box_color, boxThickness)

    lines = ["ROI {}".format(int(roi["id"]))]
    if n_detected_objs >= 0:
        lines.append("Detected: {}".format(n_detected_objs))

    put_lines_bg(
        im, lines, (x0, y0), box_color, txt_color, fontFace, fontScale, fontThickness
    )
