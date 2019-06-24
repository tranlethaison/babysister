import os
import glob
import time
import threading

import cv2 as cv
import numpy as np
import fire

from babysister.detector import YOLOv3
from babysister.tracker import SORTTracker
from babysister.babysister import detect_and_track
from babysister.utils import (
    create_unique_color_uchar, 
    putText_withBackGround, putTextWithBG,
    FPSCounter)

    
def run(
    frames_dir, rois_file='ROIs',
    input_size=[416,416], classes=['all'],
    max_boxes=100, score_thresh=0.5, iou_thresh=0.5, max_bb_size_ratio=[1,1],
    save_to=None, do_show=True, do_show_class=True
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

    # Visulization stuff
    fontFace = cv.FONT_HERSHEY_SIMPLEX
    fontScale = 0.35
    fontThickness = 1
    boxThickness = 2
    if do_show:
        winname = 'Babysister'
        cv.namedWindow(winname)
        # cv.moveWindow(winname, 0, 0)
        cv.waitKey(1)
    #--------------------------------------------------------------------------

    # ROIs
    with open(rois_file, 'r') as f:
        rois = [
            list(map(int, line.rstrip('\n').split(' ')))
            for line in f
        ]
    # There're no ROIs, create a full size one.
    if len(rois) == 0:
        frame = cv.imread(frames_path[0], cv.IMREAD_COLOR)
        frame_h, frame_w = frame.shape[:2]
        rois.append([0, 0, frame_w, frame_h])
    #--------------------------------------------------------------------------

    # Core
    # Input size is None, use frame size instead
    if input_size is None:
        frame = cv.imread(frames_path[0], cv.IMREAD_COLOR)
        input_size = reversed(frame.shape[:2])

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

    # info
    print('Processing {} images from {}'.format(len(frames_path), frames_dir))
    print('With ROIs: {}'.format(rois))
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
            classes, max_bb_size_ratio
        )
        #----------------------------------------------------------------------

        # putText Frame info
        start_tl = np.array([0, 0])
        txt = frame_path
        txt_size = putTextWithBG(
            frame, txt, start_tl,
            fontFace, 0.5, fontThickness, 
            color=(255, 255, 255), colorBG=(0, 0, 0)
        )
        print(txt)

        start_tl += [0, txt_size[1]]
        txt = 'Frame: {}'.format(frame_num)
        txt_size = putTextWithBG(
            frame, txt, start_tl,
            fontFace, 0.5, fontThickness, 
            color=(255, 255, 255), colorBG=(0, 0, 0)
        )
        print(txt)

        # fps
        start_tl += [0, txt_size[1]]
        txt = "FPS: {:.02f}".format(fpsCounter.get())
        txt_size = putTextWithBG(
            frame, txt, start_tl,
            fontFace, 0.5, fontThickness, 
            color=(255, 255, 255), colorBG=(0, 0, 0)
        )
        print(txt)

        # Go through ROIs
        detected_objs = [0] * len(rois)

        for roi_n, roi in enumerate(rois):
            # Count detected OBJs in each ROI
            roi_x, roi_y, roi_w, roi_h = roi
            for box in boxes:
                x0, y0, x1, y1 = box
                if roi_x <= (x1 + x0) / 2 <= roi_x + roi_w \
                and roi_y <= (y1 + y0) / 2 <= roi_y + roi_h:
                    detected_objs[roi_n] += 1

            # Draw ROI
            color = create_unique_color_uchar(roi_n) 
            cv.rectangle(
                frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), 
                color, boxThickness)

            # putText detected OBJs
            txt = 'Detected: {}'.format(detected_objs[roi_n])
            putTextWithBG(
                frame, txt, (roi_x, roi_y),
                fontFace, 0.5, fontThickness, 
                color=(255, 255, 255), colorBG=(0, 0, 0)
            )

        print(detected_objs)

        # Draw detections
        print('Detections:\n\tClass\tScore\tBox')
        for box, score, label in zip(boxes, scores, labels):
            color = create_unique_color_uchar(label) # (0,0,255)

            # box
            x0, y0, x1, y1 = map(int, box)
            cv.rectangle(frame, (x0,y0), (x1,y1), color, boxThickness)

            if do_show_class:
                # score
                putText_withBackGround(
                    frame, '{:.02f}'.format(score),
                    (x0,y0-20), fontFace, fontScale, fontThickness, color)

                # class
                putText_withBackGround(
                    frame, detector.classes[label],
                    (x0+40,y0-20), fontFace, fontScale, fontThickness, color)

            print('\t{}\t{}\t{}'.format(detector.classes[label], score, box))

        # Draw tracking
        print('Tracking:\n\tID\tBox')
        for track in tracks:
            id_ = int(track[4])
            color = create_unique_color_uchar(id_)

            # box
            x0, y0, x1, y1 = map(int, track[0:4])
            cv.rectangle(frame, (x0,y0), (x1,y1), color, boxThickness)

            # id_
            putText_withBackGround(
                frame, str(id_), (x0,y0), 
                fontFace, fontScale, fontThickness, color)

            print("\t{}\t{}".format(int(track[4]), track[0:4]))
        #------------------------------------------------------------------

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
        Pass in `None` to use frame/ROIs sizes. (no resizing)
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

