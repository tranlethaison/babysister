"""Example usage"""
import sys
import os
import time

import cv2 as cv
import numpy as np

from .frames_reader import FrameReadError
from .detector_core import YOLOv3TF2
#from .sort_wrapper import SORT
from .detector_1 import Detector
from .cv_logics import is_inside_roi
from .roi_manager import ROIManager
from .drawer import draw_detection, draw_tracking, draw_roi, put_line_bg
from .logger import Logger
from .fps_counter import FPSCounter
from .prompter import query_yes_no
from .time_utils import StopWatch, get_str_localtime

from .YOLOv3_TF2.yolov3_tf2.utils import draw_outputs


def run(
    framesReader,
    do_try_reading=False,
    rois_file="rois.csv",
    input_size=416,
    valid_classes=None,
    max_boxes=100,
    score_thresh=0.5,
    iou_thresh=0.5,
    max_bb_size_ratio=[1, 1],
    save_to=None,
    im_fmt="{:06d}.jpg",
    log_file=None,
    delimiter=",",
    quotechar='"',
    time_fmt="%Y/%m/%d %H:%M:%S",
    log_dist=-1,
    log_save_dist=10,
    do_show=True,
    do_show_class=True,
    winname="Babysister",
    session_config=None,
    max_uptime=-1,
    do_prompt=True,
    tiny=False,
):
    """Objects detecting, online tracking.

    Get frames by `framesReader`,
    for every gotten frame do detecting, online tracking, save result images, logs.

    Args:
        framesReader
            (:py:class:`~babysister.frames_reader.ImagesReader`
            or :py:class:`~babysister.frames_reader.VideoReader`):
            Get frames as ndarray from a source.
        do_try_reading (bool, optional):
            Whether to continue trying when `framesReader` raises
            :py:exc:`~babysister.frames_reader.FrameReadError`.
        rois_file (str, optional):
            Path to ROIs data csv file.
        input_size (list of 2 int, optional):
            Input size (width, height) use for OBJ detector initial.
            Should has same aspect ratio as frame size.
            None implies frame size.
        valid_classes (list of str or None, optional):
            Only detect these classes, omit others.
            None implies all possible classes.
            All possible classes are in "babysister/YOLOv3_TensorFlow/data/coco.names".
        max_boxes (int, optional):
            Maximum detected bounding boxes.
        score_thresh (float (from 0 to 1), optional):
            Confidence score threshold.
        iou_thresh (float (from 0 to 1), optional):
            IOU threshold use for detected boxes Non-Maximum Suppression.
        max_bb_size_ratio (list of 2 floats (from 0 to 1), optional):
            Maximum bounding box size ratio wrt frame size.
        save_to (str or None, optional):
            Path to folder that result images will be saved to.
            None implies no saving.
        im_fmt (str, optional):
            Frame name format.
        log_file (str or None, optional):
            Path to csv log file.
            None implies no logging.
        delimiter (str, optional):
            Delimiter of log file and ROIs data file.
        quotechar (str, optional):
            Quote character of log file and ROIs data file.
        time_fmt (str, optional):
            Time format of log file and ROIs data file.
        log_dist (float, optional):
            Time interval (seconds) between logs buffering, result images saving.
            Negative implies buffering all logs, saving all iamges.
        log_save_dist (float, optional):
            Time interval (seconds) between logs saving.
        do_show (bool, optional):
            Whether to show result window.
        do_show_class (bool, optional):
            Whether to show classes, confidence scores.
        winname (str, optional):
            Result window name.
        session_config (:py:class:`tensorflow.ConfigProto`, optional):
            Detector session config.
        max_uptime (float, optional):
            Maximum uptime (seconds).
            Negative implies infinite.
        do_prompt (bool, optional):
            Whether to prompt when `save_to` or `log_file` existed.
    """
    # Prepare save_to, log_file
    if not save_to:
        pass
    elif do_prompt and os.path.isdir(save_to):
        do_ow = query_yes_no(
            "{} already exist. Overwrite?".format(save_to), default="no"
        )
        if do_ow:
            pass
        else:
            print("(ʘ‿ʘ)╯ Bye!")
            sys.exit(0)
    else:
        os.makedirs(save_to, exist_ok=True)

    if not log_file:
        pass
    elif do_prompt and os.path.isfile(log_file):
        do_ow = query_yes_no(
            "{} already exist. Overwrite?".format(log_file), default="no"
        )
        if do_ow:
            pass
        else:
            print("(ʘ‿ʘ)╯ Bye!")
            sys.exit(0)
    if log_file:
        header = ["im_file_name", "timestamp", "roi_id", "n_objs"]
        logger = Logger(log_file, header, delimiter, quotechar)
        logger.open(mode="w+")
        logger.write_header()

    # Load ROIs data
    frame_w, frame_h = framesReader.get_frame_size()
    rois = ROIManager.read_rois(rois_file, delimiter, quotechar)
    if input_size is None:
        input_size = frame_w

    # [Core]
    yolov3 = YOLOv3TF2(
        size=input_size,
        score_thresh=score_thresh,
        tiny=tiny,
    )
    detector = Detector(yolov3)
    # tracker = SORT()

    # Stopwatches, FPS counter
    fpsCounter = FPSCounter(interval=1)
    log_sw = StopWatch(precision=0)
    log_save_sw = StopWatch(precision=0)

    # Init main loop conditions
    do_log_every_frame = log_dist < 0
    if not do_log_every_frame:
        n_log_writing = 0
        exp_n_log_writing = log_save_dist // log_dist

    n_save_times = 0
    exp_n_save_times = max_uptime // log_save_dist

    # Main loop
    frame_num = int(0)
    while 1:
        if frame_num == 0:
            # Start all stopwatches
            now = log_sw.start()
            log_save_sw.start_at(now)
        else:
            now = log_sw.time()  # Use this "timestamp" when writting logs

        # Read a frame
        try:
            frame = framesReader.read()
        except FrameReadError as err:
            print(err)
            if do_try_reading:
                continue
            break
        im_file_name = im_fmt.format(frame_num)

        # Whether to write log, save log
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

        # Whether to end
        if max_uptime < 0:
            do_end = False
        else:
            if do_log_save:
                n_save_times += 1
            do_end = n_save_times == exp_n_save_times

        # [Core] detecting & tracking
        boxes, scores, labels = detector.detect(frame, valid_classes, max_bb_size_ratio)
        # tracks = tracker.update(boxes, scores)

        # Count OBJs inside ROI, draw boxes
        for roi in rois:
            roi_value = (roi["x"], roi["y"], roi["w"], roi["h"])

            # >>> Process detected OBJs
            # Number of OBJs inside this ROI
            n_detected_objs = int(0)
            # Encountered objs mask
            encountered_objs_mask = np.asarray([False] * len(boxes))

            for id_ in range(len(boxes)):
                if not is_inside_roi(roi_value, boxes[id_]):
                    continue
                n_detected_objs += 1
                encountered_objs_mask[id_] = True

                draw_detection(
                    frame,
                    boxes[id_],
                    scores[id_],
                    labels[id_],
                    yolov3.class_names,
                    do_show_class,
                )

            # Filter out encountered OBJs
            if len(encountered_objs_mask) > 0:
                boxes = np.asarray(boxes)[~encountered_objs_mask]
                scores = np.asarray(scores)[~encountered_objs_mask]
                labels = np.asarray(labels)[~encountered_objs_mask]
            # <<< Process detected OBJs

            # >>> Process tracked OBJs
            # Encountered objs mask
            # encountered_objs_mask = np.asarray([False] * len(tracks))

            # for id_ in range(len(tracks)):
            #     if not is_inside_roi(roi_value, tracks[id_][:4]):
            #         continue
            #     encountered_objs_mask[id_] = True
            #     draw_tracking(frame, tracks[id_])

            # # Filter out encountered OBJs
            # if len(encountered_objs_mask) > 0:
            #     tracks = tracks[~encountered_objs_mask]
            # <<< Process tracked OBJs

            draw_roi(frame, roi, n_detected_objs)

            if do_log and log_file:
                str_time = get_str_localtime(time_fmt, now)
                log_line = [im_file_name, str_time, int(roi["id"]), n_detected_objs]
                logger.info(log_line)
                # print(log_line)

        if do_log_save and log_file:
            # str_time = get_str_localtime(time_fmt, now)
            # logger.info([None, str_time, None, None])
            logger.save()

        put_line_bg(frame, "FPS: {:.02f}".format(fpsCounter.get()), (frame_w // 2, 0))

        if do_log and save_to:
            cv.imwrite(os.path.join(save_to, im_file_name), frame)

        if do_show:
            cv.imshow(winname, frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

        if do_end:
            break

        # Restart all stopwatches
        if do_log and not do_log_every_frame:
            log_sw.start()
        if do_log_save and do_log_every_frame:
            log_save_sw.start()

        fpsCounter.tick()
        frame_num += 1

    if log_file:
        logger.close()

    if do_show:
        cv.destroyAllWindows()

    print("( ͡° ͜ʖ ͡°)_/¯ Thanks for using!")
