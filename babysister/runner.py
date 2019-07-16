"""Example usage"""
import os
import time

import cv2 as cv
import numpy as np

from .frames_reader import FrameReadError
from .yolov3_wrapper import YOLOv3
from .sort_wrapper import SORT
from .detector import Detector
from .cv_logics import is_inside_roi
from .roi_manager import ROIManager 
from .drawer import draw_detection, draw_tracking, draw_roi, put_line_bg
from .logger import Logger
from .fps_counter import FPSCounter
from .prompter import query_yes_no
from .time_utils import StopWatch, get_str_localtime

    
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
    log_file=None, 
    delimiter=',', 
    quotechar='"',
    time_fmt='%Y/%m/%d %H:%M:%S',
    log_dist=-1, 
    log_save_dist=10,
    do_show=True, 
    do_show_class=True,
    winname="Babysister",
    session_config=None,
    do_detect=True,
    max_uptime=-1
):
    """"""
    if save_to:
        if os.path.isdir(save_to):
            do_ow = query_yes_no(
                '{} already exist. Overwrite?'.format(save_to), default="no")
            if do_ow:
                pass
            else:
                print("(ʘ‿ʘ)╯ Bye!")
                exit(0)
        else:
            os.makedirs(save_to)

    if log_file:
        if os.path.isfile(log_file):
            do_ow = query_yes_no(
                '{} already exist. Overwrite?'.format(log_file), default="no")
            if do_ow:
                pass
            else:
                print("(ʘ‿ʘ)╯ Bye!")
                exit(0)

        header = ['im_file_name', 'timestamp', 'roi_id', 'n_objs']
        logger = Logger(log_file, header, delimiter, quotechar)
        logger.write_header()
    # -------------------------------------------------------------------------

    frame_w, frame_h = framesReader.get_frame_size()

    rois = ROIManager.read_rois(rois_file, delimiter, quotechar)

    if input_size is None:
        input_size = [frame_w, frame_h]

    if do_detect:
        # Core
        yolov3 = YOLOv3(
            input_size[::-1], max_boxes, score_thresh, iou_thresh,
            session_config=session_config)
        detector = Detector(yolov3)

        tracker = SORT()
    # << Core 
    # -------------------------------------------------------------------------

    print("Detecting and tracking. Press 'q' at {} to quit.".format(winname))
    fpsCounter = FPSCounter(limit=1)
    log_sw = StopWatch(precision=0)
    log_save_sw = StopWatch(precision=0)
    uptime_sw = StopWatch(precision=0)

    do_log_every_frame = log_dist < 0
    n_log_writing = 0
    exp_n_log_writing = log_save_dist // log_dist

    frame_num = int(0)
    while 1:
        try:
            frame = framesReader.read()
        except FrameReadError as err:
            print(err)
            if do_try_reading:
                continue
            break
        im_file_name = im_format.format(frame_num)
        
        if frame_num == 0:
            now = log_sw.start()
            log_save_sw.start_at(now)
            uptime_sw.start_at(now)
        else:
            now = log_sw.time()

        if do_log_every_frame:
            do_log = True
            do_log_save = log_save_sw.elapsed() >= log_save_dist
        else:
            do_log = log_sw.elapsed() >= log_dist
            if do_log:
                n_log_writing += 1

            do_log_save = n_log_writing == exp_n_log_writing 
            if do_log_save:
                n_log_writing = 0

        if max_uptime < 0:
            do_end = False
        else:
            do_end = uptime_sw.elapsed() >= max_uptime

        if do_detect:
            boxes, scores, labels = \
                detector.detect(frame, valid_classes, max_bb_size_ratio)
            tracks = tracker.update(boxes, scores) 

        for roi in rois:
            roi_value = (roi['x'], roi['y'], roi['w'], roi['h'])

            if do_detect:
                # Keep track of
                n_detected_objs = int(0)  # number of objs inside this ROI

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
            else:
                n_detected_objs = int(-1)  # number of objs inside this ROI

            draw_roi(frame, roi, n_detected_objs)

            if do_log and log_file: 
                str_time = get_str_localtime(time_fmt, now)
                log_line = \
                    [im_file_name, str_time, int(roi['id']), n_detected_objs]
                logger.info(log_line)
                print(log_line)

        if do_log:
            if not do_log_every_frame:
                log_sw.start()

        if do_log_save and log_file:
            str_time = get_str_localtime(time_fmt, now)
            # logger.info([None, str_time, None, None])
            logger.save()
            if do_log_every_frame:
                log_save_sw.start()
            print("log saved at", str_time)

        put_line_bg(
            frame, "FPS: {:.02f}".format(fpsCounter.get()), (frame_w//2, 0))

        if do_log and save_to:
            cv.imwrite(os.path.join(save_to, im_file_name), frame)

        if do_show:
            cv.imshow(winname, frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        if do_end:
            str_time = get_str_localtime(time_fmt, now)
            print("max_uptime elapsed, end process at", str_time)
            break

        fpsCounter.tick()
        frame_num += 1

    if log_file:
        logger.close()

    if do_show:
        cv.destroyAllWindows()
    
    print("( ͡° ͜ʖ ͡°)_/¯ Thanks for using!")

