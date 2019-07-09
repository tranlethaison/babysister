""""""
import os

import fire
import cv2 as cv

from babysister.prompter import query_yes_no
from babysister.roi_manager import select_rois_over_image


def select_rois(
    in_file, is_video=False, save_to='rois.csv',
    delimiter=',', quotechar="'"
):
    """"""
    if os.path.isfile(save_to):
        do_ow = query_yes_no(
            'File exists: {}.\n Overwrite?'.format(save_to), default="no")
        if do_ow: 
            pass
        else:
            print('Ok, thanks. Bye')
            exit(1)

    if is_video:
        cap = cv.VideoCapture(in_file)
        assert cap.isOpened()
        ret, im = cap.read()
        assert ret, "Can't receive frame"
    else:
        im = cv.imread(in_file)

    select_rois_over_image(im, save_to, delimiter, quotechar)


if __name__ == "__main__":
    fire.Fire(select_rois)

