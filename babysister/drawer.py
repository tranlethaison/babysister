import numpy as np
import cv2 as cv
from .utils import create_unique_color_uchar, putTextWithBG


def draw_detection(
    img, box, score, label, classes,
    fontFace, fontScale, fontThickness, boxThickness,
    do_show_class=True
):
    color = create_unique_color_uchar(label)

    # box
    x0, y0, x1, y1 = map(int, box)
    cv.rectangle(img, (x0,y0), (x1,y1), color, boxThickness)

    if do_show_class:
        # score
        txt = '{:.02f}'.format(score)

        (txt_w, txt_h), baseLine = \
            cv.getTextSize(txt, fontFace, fontScale, fontThickness)
        top_left = np.array([x0, y0 - txt_h - baseLine])

        (txt_w, txt_h), baseLine = putTextWithBG(
            img, txt, top_left,
            fontFace, fontScale, fontThickness, 
            color=(255, 255, 255), colorBG=color)

        # class
        txt = classes[label]
        top_left += [txt_w + 2, 0]
        putTextWithBG(
            img, txt, top_left,
            fontFace, fontScale, fontThickness, 
            color=(255, 255, 255), colorBG=color)

        print('\t{}\t{}\t{}'.format(classes[label], score, box))


def draw_tracking(
    img, track,
    fontFace, fontScale, fontThickness, boxThickness
):
    id_ = int(track[4])
    color = create_unique_color_uchar(id_)

    # box
    x0, y0, x1, y1 = map(int, track[:4])
    cv.rectangle(img, (x0,y0), (x1,y1), color, boxThickness)

    # id_
    putTextWithBG(
        img, str(id_), (x0,y0),
        fontFace, fontScale, fontThickness, 
        color=(255, 255, 255), colorBG=color)

    print("\t{}\t{}".format(int(track[4]), track[:4]))

