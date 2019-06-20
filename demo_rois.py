import os
import glob
import time
import fire
import cv2 as cv
import numpy as np
from pprint import pprint

from babysister.detector import YOLOv3
from babysister.tracker import SORTTracker
from babysister.utils import (
    create_unique_color_uchar, 
    putText_withBackGround,
    FPSCounter)


def detect_and_track(frame, rois, classes, max_bb_size_ratio):
    '''
    '''
    # detect and track result, each ROI will get it own
    dt_results = [None] * len(rois['value'])

    for roi_num, roi_value in enumerate(rois['value']):
        # ROI data
        x, y, w, h = roi_value
        input_size = rois['input_sizes'][roi_num]
        detector = rois['detectors'][roi_num]
        tracker = rois['trackers'][roi_num]
        roi = frame[y:y+h, x:x+w]

        # input data
        input_data = cv.resize(
            roi, dsize=tuple(input_size), interpolation=cv.INTER_LANCZOS4)
        input_data = cv.cvtColor(input_data, cv.COLOR_BGR2RGB)
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

        # detect
        boxes, scores, labels = detector.detect(input_data)

        # filter by class
        if 'all' not in classes:
            tmp_boxes, tmp_scores, tmp_labels = [], [], []
            for box, score, label in zip(boxes, scores, labels):
                if detector.classes[label] in classes:
                    tmp_boxes.append(box)
                    tmp_scores.append(score)
                    tmp_labels.append(label)
            boxes, scores, labels = np.array(tmp_boxes), tmp_scores, tmp_labels

        # rescale boxes
        if boxes.shape[0] > 0:
            size_ratio = np.divide([w, h], input_size)
            boxes[:,0] *= size_ratio[0]
            boxes[:,1] *= size_ratio[1]
            boxes[:,2] *= size_ratio[0]
            boxes[:,3] *= size_ratio[1]

        #filter by box size wrt image size.
        if np.greater([1,1], max_bb_size_ratio).any():
            tmp_boxes, tmp_scores, tmp_labels = [], [], []
            for box, score, label in zip(boxes, scores, labels):
                x0, y0, x1, y1 = box
                size_ratio = np.divide([x1-x0, y1-y0], [frame_w, frame_h])

                if np.greater(size_ratio, max_bb_size_ratio).any():
                    continue

                tmp_boxes.append(box)
                tmp_scores.append(score)
                tmp_labels.append(label)
            boxes, scores, labels = np.array(tmp_boxes), tmp_scores, tmp_labels

        # track
        tracks = tracker.update(boxes, scores)

        dt_results[roi_num] = [boxes, scores, labels], tracks
    
    return dt_results
    

def run(
    frames_dir, rois_file='ROIs',
    input_size=None, classes=['all'],
    max_boxes=100, score_thresh=0.5, iou_thresh=0.5, max_bb_size_ratio=[1,1],
    save_to=None, do_show=True, do_show_class=True
):
    # Frames sequence
    frames_path = sorted(glob.glob(frames_dir + '/*.jpg'))
    assert len(frames_path) > 0

    # ROIs
    rois = {}
    with open(rois_file, 'r') as f:
        rois['value'] = [
            list(map(int, line.rstrip('\n').split(' ')))
            for line in f
        ]

    # Save prepare
    if save_to:
        if os.path.isdir(save_to):
            do_replace = input('Save directory already exist. Overwrite? y/N\n')
            if do_replace.lower() == 'y':
                pass
            else:
                print('OK. Thank You.')
                exit(0)
        else:
            os.makedirs(save_to)

    # Each ROI get it own 
    # Input size
    rois['input_sizes'] = [None] * len(rois['value']) 

    # Detector 
    yolov3_data_d = 'babysister/YOLOv3_TensorFlow/data'
    anchor_path = os.path.join(yolov3_data_d, 'yolo_anchors.txt')
    class_name_path = os.path.join(yolov3_data_d, 'coco.names')
    restore_path = os.path.join(yolov3_data_d, 'darknet_weights/yolov3.ckpt')
    rois['detectors'] = [None] * len(rois['value']) 

    # Tracker
    rois['trackers'] = [None] * len(rois['value'])

    for roi_num, roi_value in enumerate(rois['value']):
        # Detector
        x, y, w, h = roi_value
        rois['input_sizes'][roi_num] = input_size or [w, h]

        rois['detectors'][roi_num] = YOLOv3(
            reversed(rois['input_sizes'][roi_num]),
            anchor_path, class_name_path, restore_path,
            max_boxes, score_thresh, iou_thresh)

        # Tracker
        rois['trackers'][roi_num] = SORTTracker()

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

    # info
    print('Processing {} images from {}'.format(len(frames_path), frames_dir))
    print('With ROIs: {}'.format(rois))
    print('YOLOv3')
    print('Max boxes: {}\nScore threshold: {}\nIOU threshold: {}'
        .format(max_boxes, score_thresh, iou_thresh))
    print('Result images will be saved to {}\n'.format(save_to))

    # fps
    fpsCounter = FPSCounter(limit=1)

    for frame_num, frame_path in enumerate(frames_path):
        # Frame info
        frame_info = "{}\nFrame: {}".format(frame_path, frame_num)
        print(frame_info)

        # Read
        frame = cv.imread(frame_path, cv.IMREAD_COLOR)

        # Detect and track for each ROI
        dt_results = detect_and_track(frame, rois, classes, max_bb_size_ratio)

        # Drawing
        for roi_num, (roi_value, dt_result) \
        in enumerate(zip(rois['value'], dt_results)):
            x, y, w, h = roi_value
            detector = rois['detectors'][roi_num]
            [boxes, scores, labels], tracks = dt_result

            # Draw ROI
            color = create_unique_color_uchar(roi_num) 
            cv.rectangle(frame, (x, y), (x+w, y+h), color, boxThickness)

            # putText
            frame[y:y+120, x:x+150, 1] = 255
            text_x, text_y = 5 + x, 25 + y
            text_line_gap = 20

            # frame_info
            for i, line in enumerate(frame_info.split("\n")):
                text_y += i * text_line_gap
                cv.putText(
                    frame, line, (text_x, text_y),
                    fontFace, 0.5, (0,0,0), fontThickness)

            # fps
            str_fps = "FPS: {:.02f}".format(fpsCounter.get())
            text_y += text_line_gap
            cv.putText(
                    frame, str_fps, (text_x, text_y),
                    fontFace, 0.5, (0,0,0), fontThickness)
            print(str_fps)

            # counts
            str_counts = \
                "Detected: {}\nTracked:  {}".format(len(labels), len(tracks))
            text_y += text_line_gap
            for i, line in enumerate(str_counts.split('\n')):
                text_y += i * text_line_gap
                cv.putText(
                    frame, line, (text_x, text_y),
                    fontFace, 0.5, (0,0,0), fontThickness)
            print(str_counts)

            # draw detections
            print('Detections:\n\tClass\tScore\tBox')
            for box, score, label in zip(boxes, scores, labels):
                color = create_unique_color_uchar(label) # (0,0,255)

                # box
                x0, y0, x1, y1 = map(int, box + [x, y, x, y])
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

            # draw tracking
            print('Tracking:\n\tID\tBox')
            for track in tracks:
                id_ = int(track[4])
                color = create_unique_color_uchar(id_)

                # box
                x0, y0, x1, y1 = map(int, track[0:4] + [x, y, x, y])
                cv.rectangle(frame, (x0,y0), (x1,y1), color, boxThickness)

                # id_
                putText_withBackGround(
                    frame, str(id_), (x0,y0), 
                    fontFace, fontScale, fontThickness, color)

                print("\t{}\t{}".format(int(track[4]), track[0:4]))

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
    print("""
Objects detection and online tracking with multiple ROIs.

Usage:

Descriptions:
    """)


if __name__ == '__main__':
    fire.Fire()

