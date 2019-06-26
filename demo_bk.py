import os
import time
import glob

import cv2 as cv
import numpy as np
import fire

from babysister.detector import YOLOv3
from babysister.tracker import SORTTracker
from babysister.babysister import detect_and_track
from babysister.roi_manager import fieldnames, read_rois
from babysister.utils import (
    create_unique_color_uchar, 
    putTextWithBG,
    FPSCounter)
from babysister.logger import Logger

    
def run(
    frames_dir, rois_file='rois.csv',
    input_size=[416,416], classes=['all'],
    max_boxes=100, score_thresh=0.5, iou_thresh=0.5, max_bb_size_ratio=[1,1],
    save_to=None, log_file='log.cvs',
    do_show=True, do_show_class=True
):
    # Frames sequence
    frames_path = sorted(glob.glob(frames_dir + '/*.jpg'))
    assert len(frames_path) > 0

    # Save prepare
    if save_to:
        if os.path.isdir(save_to):
            do_replace = \
                input('{} already exist. Overwrite? y/N\n'.format(save_to))
            if do_replace.lower() == 'y':
                pass
            else:
                print('OK. Thank You.')
                exit(0)
        else:
            os.makedirs(save_to)

    # Log
    field_names = ['roi_id', 'n_objs', 'timestamp']
    time_fmt = '%Y/%m/%d %H:%M:%S'
    timestamp = time.time()
    log_dist = 5
    logger = Logger(field_names, save_to=log_file, delimiter=',', quotechar="'")
    logger.info(field_names)
    #--------------------------------------------------------------------------

    # ROIs
    rois = read_rois(rois_file, delimiter=',', quotechar="'")
    # There're no ROIs, create one with size of the whole frame.
    if len(rois) == 0:
        frame = cv.imread(frames_path[0], cv.IMREAD_COLOR)
        frame_h, frame_w = frame.shape[:2]

        values = [0, 0, 0, frame_w, frame_h, -1]
        rois = [{}]
        for fieldname, value in zip(fieldnames, values):
            rois[0][fieldname] = value
    #--------------------------------------------------------------------------

    # Core
    if len(input_size) == 0:
        # use frame size instead
        frame = cv.imread(frames_path[0], cv.IMREAD_COLOR)
        input_size = list(reversed(frame.shape[:2]))

    # Detector
    yolov3_data_d = 'babysister/YOLOv3_TensorFlow/data'
    anchor_path = os.path.join(yolov3_data_d, 'yolo_anchors.txt')
    class_name_path = os.path.join(yolov3_data_d, 'coco.names')
    restore_path = os.path.join(yolov3_data_d, 'darknet_weights/yolov3.ckpt')

    detector = YOLOv3(
        reversed(input_size),
        anchor_path, class_name_path, restore_path,
        max_boxes, score_thresh, iou_thresh)

    # Tracker
    tracker = SORTTracker()
    #--------------------------------------------------------------------------

    # Visulization stuff
    fontFace = cv.FONT_HERSHEY_SIMPLEX
    fontScale = 0.35
    fontThickness = 1
    boxThickness = 2
    if do_show:
        winname = 'Babysister {}'.format(input_size)
        cv.namedWindow(winname)
        # cv.moveWindow(winname, 0, 0)
        cv.waitKey(1)

    # info
    print('Processing {} images from {}'.format(len(frames_path), frames_dir))
    print('With ROIs:\n{}'.format(rois))
    print('YOLOv3')
    print('Max boxes: {}\nScore threshold: {}\nIOU threshold: {}'
        .format(max_boxes, score_thresh, iou_thresh))
    print('Result images will be saved to {}\n'.format(save_to))

    # fps
    fpsCounter = FPSCounter(limit=1)

    # Go through each frame
    for frame_num, frame_path in enumerate(frames_path):
        # Read image
        frame = cv.imread(frame_path, cv.IMREAD_COLOR)

        # Detect and track
        boxes, scores, labels, tracks = detect_and_track(
            frame, 
            input_size, detector, tracker,
            classes, max_bb_size_ratio)
        #----------------------------------------------------------------------

        # Draw detections
        print('Detections:\n\tClass\tScore\tBox')
        #for box, score, label in zip(boxes, scores, labels):
        #    color = create_unique_color_uchar(label) # (0,0,255)

        #    # box
        #    x0, y0, x1, y1 = map(int, box)
        #    cv.rectangle(frame, (x0,y0), (x1,y1), color, boxThickness)

        #    if do_show_class:
        #        # score
        #        txt = '{:.02f}'.format(score)

        #        (txt_w, txt_h), baseLine = \
        #            cv.getTextSize(txt, fontFace, fontScale, fontThickness)
        #        top_left = np.array([x0, y0 - txt_h - baseLine])

        #        (txt_w, txt_h), baseLine = putTextWithBG(
        #            frame, txt, top_left,
        #            fontFace, fontScale, fontThickness, 
        #            color=(255, 255, 255), colorBG=color)

        #        # class
        #        txt = detector.classes[label]
        #        top_left += [txt_w + 2, 0]
        #        putTextWithBG(
        #            frame, txt, top_left,
        #            fontFace, fontScale, fontThickness, 
        #            color=(255, 255, 255), colorBG=color)

        #    print('\t{}\t{}\t{}'.format(detector.classes[label], score, box))

        # Draw tracking
        print('Tracking:\n\tID\tBox')
        for track in tracks:
            id_ = int(track[4])
            color = create_unique_color_uchar(id_)

            # box
            x0, y0, x1, y1 = map(int, track[0:4])
            #cv.rectangle(frame, (x0,y0), (x1,y1), color, boxThickness)

            # id_
            putTextWithBG(
                frame, str(id_), (x0,y0),
                fontFace, fontScale, fontThickness, 
                color=(255, 255, 255), colorBG=color)

            print("\t{}\t{}".format(int(track[4]), track[0:4]))
        #----------------------------------------------------------------------

        # putText Frame info
        top_left = np.array([0, 0])
        txt = frame_path
        (txt_w, txt_h), baseLine = putTextWithBG(
            frame, txt, top_left,
            fontFace, fontScale, fontThickness, 
            color=(255, 255, 255), colorBG=(0, 0, 0))
        print(txt)

        top_left += [0, txt_h + baseLine]
        txt = 'Frame: {}'.format(frame_num)
        (txt_w, txt_h), baseLine = putTextWithBG(
            frame, txt, top_left,
            fontFace, fontScale, fontThickness, 
            color=(255, 255, 255), colorBG=(0, 0, 0))
        print(txt)

        # fps
        top_left += [0, txt_h + baseLine]
        txt = "FPS: {:.02f}".format(fpsCounter.get())
        (txt_w, txt_h), baseLine = putTextWithBG(
            frame, txt, top_left,
            fontFace, fontScale, fontThickness, 
            color=(255, 255, 255), colorBG=(0, 0, 0))
        print(txt)

        # Go through ROIs
        detected_objs = [0] * len(rois)
        is_full = [False] * len(rois)

        # Log
        now = time.time()
        do_log = now - timestamp >= log_dist

        for roi_n, roi in enumerate(rois):
            # Count detected OBJs in each ROI
            # If an obj centroid is inside a ROI, it also is.
            for box, score, label in zip(boxes, scores, labels):
                x0, y0, x1, y1 = map(int, box)
                if roi['x'] <= (x1 + x0) / 2 <= roi['x'] + roi['w'] \
                and roi['y'] <= (y1 + y0) / 2 <= roi['y'] + roi['h']:
                    detected_objs[roi_n] += 1

                    color = create_unique_color_uchar(label) # (0,0,255)

                    # box
                    cv.rectangle(frame, (x0,y0), (x1,y1), color, boxThickness)

                    if do_show_class:
                        # score
                        txt = '{:.02f}'.format(score)

                        (txt_w, txt_h), baseLine = \
                            cv.getTextSize(txt, fontFace, fontScale, fontThickness)
                        top_left = np.array([x0, y0 - txt_h - baseLine])

                        (txt_w, txt_h), baseLine = putTextWithBG(
                            frame, txt, top_left,
                            fontFace, fontScale, fontThickness, 
                            color=(255, 255, 255), colorBG=color)

                        # class
                        txt = detector.classes[label]
                        top_left += [txt_w + 2, 0]
                        putTextWithBG(
                            frame, txt, top_left,
                            fontFace, fontScale, fontThickness, 
                            color=(255, 255, 255), colorBG=color)

                    print('\t{}\t{}\t{}'.format(detector.classes[label], score, box))

            # Determine if ROI is full
            is_full[roi_n] = \
                roi['max_objects'] >= 0 \
                and detected_objs[roi_n] >= roi['max_objects']

            # Draw ROI
            color = (0, 0, 0) #create_unique_color_uchar(roi_n) 
            cv.rectangle(
                frame, 
                (roi['x'], roi['y']), 
                (roi['x'] + roi['w'], roi['y'] + roi['h']), 
                color, 3)

            # putText detected OBJs
            txt = 'Detected: {}'.format(detected_objs[roi_n])
            top_left = np.array([roi['x'], roi['y']])
            (txt_w, txt_h), baseLine = putTextWithBG(
                frame, txt, top_left,
                fontFace, fontScale, fontThickness, 
                color=(255, 255, 255), colorBG=(0, 0, 0))

            # putText is ROI full
            txt = 'Max objects: {}'.format(roi['max_objects'])
            top_left += [0, txt_h + baseLine]
            (txt_w, txt_h), baseLine = putTextWithBG(
                frame, txt, top_left,
                fontFace, fontScale, fontThickness, 
                color=(255, 255, 255), colorBG=(0, 0, 0))

            txt = 'Is full: {}'.format(is_full[roi_n])
            top_left += [0, txt_h + baseLine]
            (txt_w, txt_h), baseLine = putTextWithBG(
                frame, txt, top_left,
                fontFace, fontScale, fontThickness, 
                color=(255, 255, 255), colorBG=(0, 0, 0))

            # Log
            if do_log: 
                timestamp = now
                logger.info([
                    roi['id'], 
                    detected_objs[roi_n], 
                    time.strftime(time_fmt, time.localtime(timestamp))])
        #----------------------------------------------------------------------

        # save
        if save_to:
            _, frame_name = os.path.split(frame_path)
            result_frame_path = os.path.join(save_to, frame_name)
            cv.imwrite(result_frame_path, frame)

        # show
        if do_show:
            cv.imshow(winname, frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        # fps
        fpsCounter.tick()
        print(flush=True)

    logger.close()
    if do_show:
        cv.destroyAllWindows()


def help():
    print(r"""
Objects detection and online tracking with multiple ROIs.

Usage:
demo.py run \
    FRAMES_DIR [ROIS_FILE] \
    [INPUT_SIZE] [CLASSES] \
    [MAX_BOXES] [SCORE_THRESH] [IOU_THRESH] [MAX_BB_SIZE_RATIO] \
    [SAVE_TO] [DO_SHOW] [DO_SHOW_CLASS]

demo.py run \
    --frames-dir FRAMES_DIR [--rois-file ROIS_FILE] \
    [--input-size INPUT_SIZE] [--classes CLASSES] \
    [--max-boxes MAX_BOXES] [--score-thresh SCORE_THRESH] \
    [--iou-thresh IOU_THRESH] [--max-bb-size-ratio MAX_BB_SIZE_RATIO] \
    [--save-to SAVE_TO] \
    [--do-show DO_SHOW] [--do-show-class DO_SHOW_CLASS]

Descriptions:
    --frames-dir <string>
        Directory that contain sequences of frames (jpeg).

    --rois-file <string>
        Path to ROIs file (created manualy, or with `select_rois.py`).
        If do not want to use ROIs, pass in an empty file.

        ROI contains: 
            top-left coordinate, width, height
        ROI format: 
            x y width height
        ROIs file contains ROI, each on 1 line. 

    --input-size <2-tuple>
        YOLOv3 input size.
        Pass in `None` to use frame sizes. (no resizing)
        Default: [416,416]

    --classes <list of string>
        list of classes used for filterring.
        Default: "['all']" (no filter)

    --max-boxes <integer>
        maximum number of predicted boxes you'd like.
        Default: 100

    --score-thresh <float, in range [0, 1]>
        if [ highest class probability score < score threshold]
            then get rid of the corresponding boxes
        Default: 0.5

    --iou-thresh <float, in range [0, 1]>
        "intersection over union" threshold used for NMS filtering.
        Default: 0.5

    --max_bb_size_ratio <2-tuple, in range [0, 0] ~ [1, 1]>
        Boxes maximum size ratio wrt frame size.
        Default: [1,1]

    --save-to <string>
        Directory to save result images.
        Default: not to save

    --do-show <boolean>
        Whether to display result.
        Default: True (1)

    --do-show-class <boolean>
        Whether to display classes, scores.
        Default: True (1)
    """)


if __name__ == '__main__':
    fire.Fire()

