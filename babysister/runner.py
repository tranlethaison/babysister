"""Main process"""
import os
import time

import cv2 as cv
import numpy as np

from .frames_reader import FrameReadError
from .yolov3_wrapper import YOLOv3
from .sort_wrapper import SORT
from .detector import Detector
from .cv_logics import is_inside_roi
from .roi_manager import create_roi, read_rois
from .drawer import draw_detection, draw_tracking, draw_roi, put_line_bg
from .logger import Logger
from .fps_counter import FPSCounter
from .prompter import query_yes_no

    
def run(
    framesReader, 
    do_try_reading=False,
    rois_file='rois.csv',
    input_size=[416,416], 
    valid_classes=None,
    max_boxes=100, 
    score_thresh=0.5, 
    iou_thresh=0.5, 
    max_bb_size_ratio=[1,1],
    save_to=None, 
    im_format="{:06d}.jpg",
    log_file='log.cvs', 
    delimiter=',', 
    quotechar='"',
    log_dist=10, 
    log_save_dist=60,
    do_show=True, 
    do_show_class=True,
    winname="BabySister",
    session_config=None
):
    """"""
    if save_to:
        if os.path.isdir(save_to):
            do_ow = query_yes_no(
                '{} already exist. Overwrite?'.format(save_to), default="no")
            if do_ow:
                pass
            else:
                print("(ʘ‿ʘ)╯ Bye")
                exit(0)
        else:
            os.makedirs(save_to)

    if os.path.isfile(log_file):
        do_ow = query_yes_no(
            '{} already exist. Overwrite?'.format(log_file), default="no")
        if do_ow:
            pass
        else:
            print("(ʘ‿ʘ)╯ Bye")
            exit(0)
    # -------------------------------------------------------------------------

    frame_w, frame_h = framesReader.get_frame_size()

    rois = read_rois(rois_file, delimiter, quotechar)

    if input_size is None:
        input_size = [frame_w, frame_h]

    # Core
    yolov3 = YOLOv3(
        input_size[::-1], max_boxes, score_thresh, iou_thresh,
        session_config=session_config)
    detector = Detector(yolov3)

    tracker = SORT()
    # << Core 
    # -------------------------------------------------------------------------

    header = ['id', 'n_objs', 'timestamp']
    logger = Logger(log_file, header, delimiter, quotechar)
    logger.write_header()
    log_seconds = log_save_seconds = time.time()

    fpsCounter = FPSCounter(limit=1)

    print("Detecting and tracking. Press 'q' at {} to quit.".format(winname))
    frame_num = int(0)
    while 1:
        try:
            frame = framesReader.read()
        except FrameReadError as err:
            print(err)
            if do_try_reading:
                continue
            break

        boxes, scores, labels = \
            detector.detect(frame, valid_classes, max_bb_size_ratio)

        tracks = tracker.update(boxes, scores) 

        now = time.time()
        do_log = now - log_seconds >= log_dist
        do_log_save = now - log_save_seconds >= log_save_dist

        for roi in rois:
            roi_value = (roi['x'], roi['y'], roi['w'], roi['h'])

            # Keep track of
            n_detected_objs = 0  # number of objs inside this ROI

            # Encountered objs mask, for filter out later
            encountered_objs_mask = np.asarray([False] * len(boxes))
            for id_ in range(len(boxes)):
                if not is_inside_roi(roi_value, boxes[id_]):
                    continue

                n_detected_objs += 1
                encountered_objs_mask[id_] = True
                draw_detection(
                    frame,
                    boxes[id_], scores[id_], labels[id_], yolov3.classes,
                    do_show_class)

            if len(encountered_objs_mask) > 0:
                boxes = np.asarray(boxes)[~encountered_objs_mask]
                scores = np.asarray(scores)[~encountered_objs_mask]
                labels = np.asarray(labels)[~encountered_objs_mask]
            # -----------------------------------------------------------------

            encountered_objs_mask = np.asarray([False] * len(tracks))
            for id_ in range(len(tracks)):
                if not is_inside_roi(roi_value, tracks[id_][:4]):
                    continue

                encountered_objs_mask[id_] = True
                draw_tracking(frame, tracks[id_])

            if len(encountered_objs_mask) > 0:
                tracks = tracks[~encountered_objs_mask]
            # -----------------------------------------------------------------

            draw_roi(frame, roi, n_detected_objs)

            if do_log or frame_num == 0: 
                log_seconds = now
                logger.info([roi['id'], n_detected_objs, log_seconds])

            if do_log_save:
                log_save_seconds = now
                logger.save()

        put_line_bg(
            frame, "FPS: {:.02f}".format(fpsCounter.get()), (frame_w//2, 0))

        if save_to:
            cv.imwrite(
                os.path.join(save_to, im_format.format(frame_num)), 
                frame)

        if do_show:
            cv.imshow(winname, frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        fpsCounter.tick()
        frame_num += 1

    logger.close()

    if do_show:
        cv.destroyAllWindows()
    
    print("( ͡° ͜ʖ ͡°)_/¯ Thanks for using")

