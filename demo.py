"""Demo"""
import os
import time
import glob

import cv2 as cv
import numpy as np
import fire

from babysister.frames_reader import FramesReader, VideoReader
from babysister.yolov3_wrapper import YOLOv3
from babysister.sort_wrapper import SORT
from babysister.detector import Detector
from babysister.cv_logics import is_inside_roi
from babysister.roi_manager import create_roi, read_rois
from babysister.drawer import (
    draw_detection, draw_tracking, draw_roi, put_line_bg)
from babysister.logger import Logger
from babysister.fps_counter import FPSCounter
from babysister.prompter import query_yes_no

    
def _run(
    frames_reader, 
    rois_file='rois.csv',
    input_size=[416,416], valid_classes=['all'],
    max_boxes=100, score_thresh=0.5, iou_thresh=0.5, max_bb_size_ratio=[1,1],
    save_to=None, do_show=True, do_show_class=True,
    log_file='log.cvs'
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

    frame_w, frame_h = frames_reader.get_frame_size()

    rois = read_rois(rois_file, delimiter=',', quotechar="'")
    if len(rois) == 0:
        values = [0, 0, 0, frame_w, frame_h, -1]
        rois = [create_roi(values)]

    if input_size is None:
        input_size = [frame_w, frame_h]

    # Core
    yolov3 = YOLOv3(input_size[::-1], max_boxes, score_thresh, iou_thresh)
    detector = Detector(yolov3)

    tracker = SORT()
    # << Core 

    if do_show:
        winname = 'Babysister {}'.format(input_size)
        cv.namedWindow(winname)
        # cv.moveWindow(winname, 0, 0)
        cv.waitKey(1)

    log_dist = 5  # log distance (seconds)
    seconds = time.time()
    logger = Logger(save_to=log_file, delimiter=',', quotechar="'")
    logger.write_header()

    fpsCounter = FPSCounter(limit=1)

    for frame_num, frame in enumerate(frames_reader.read()):
        boxes, scores, labels = \
            detector.detect(frame, valid_classes, max_bb_size_ratio)

        tracks = tracker.update(boxes, scores) 

        now = time.time()
        do_log = now - seconds >= log_dist

        for roi in rois:
            roi_value = (roi['x'], roi['y'], roi['w'], roi['h'])

            # Keep track of
            n_detected_objs = 0  # number of objs inside this ROI
            is_full = False  # is this ROI full

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

            is_full = ( 
                roi['max_objects'] >= 0 
                and n_detected_objs >= roi['max_objects'])

            encountered_objs_mask = np.asarray([False] * len(tracks))
            for id_ in range(len(tracks)):
                if not is_inside_roi(roi_value, tracks[id_][:4]):
                    continue

                encountered_objs_mask[id_] = True
                draw_tracking(frame, tracks[id_])

            if len(encountered_objs_mask) > 0:
                tracks = tracks[~encountered_objs_mask]

            draw_roi(frame, roi, n_detected_objs, is_full)

            if do_log or frame_num == 0: 
                seconds = now
                logger.info([roi['id'], n_detected_objs, seconds])

        put_line_bg(
            frame, "FPS: {:.02f}".format(fpsCounter.get()), (frame_w-128,0))

        if save_to:
            cv.imwrite(
                os.path.join(save_to, "{:06d}.jpg".format(frame_num)), 
                frame)

        if do_show:
            cv.imshow(winname, frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        fpsCounter.tick()

    logger.close()

    if do_show:
        cv.destroyAllWindows()
    
    print("( ͡° ͜ʖ ͡°)_/¯ Thanks for using")


def frames(
    frames_dir,
    rois_file='rois.csv',
    input_size=[416,416], valid_classes=['all'],
    max_boxes=100, score_thresh=0.5, iou_thresh=0.5, max_bb_size_ratio=[1,1],
    save_to=None, do_show=True, do_show_class=True,
    log_file='log.cvs'
):
    """Demo for sequence of frames"""
    frames_reader = FramesReader(frames_dir, "jpg")
    _run(
        frames_reader,
        rois_file,
        input_size, valid_classes,
        max_boxes, score_thresh, iou_thresh, max_bb_size_ratio,
        save_to, do_show, do_show_class,
        log_file)


def video(
    video_path,
    rois_file='rois.csv',
    input_size=[416,416], valid_classes=['all'],
    max_boxes=100, score_thresh=0.5, iou_thresh=0.5, max_bb_size_ratio=[1,1],
    save_to=None, do_show=True, do_show_class=True,
    log_file='log.cvs'
):
    """Demo for video"""
    frames_reader = VideoReader(video_path)
    _run(
        frames_reader,
        rois_file,
        input_size, valid_classes,
        max_boxes, score_thresh, iou_thresh, max_bb_size_ratio,
        save_to, do_show, do_show_class,
        log_file)


if __name__ == '__main__':
    fire.Fire()

