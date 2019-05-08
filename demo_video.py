import os
import glob
import time
import fire
import cv2 as cv
import numpy as np
from pprint import pprint

from babysister.detector import YOLOv3
from babysister.tracker import SORTTracker
from babysister.utils import create_unique_color_uchar, putText_withBackGround


def run(
    video_path, input_size=[416,416],
    classes=['all'],
    max_boxes=100, score_thresh=0.5, iou_thresh=0.5, max_bb_size_ratio=[1,1],
    save_to=None, fourcc='XVID', do_show=True, do_show_class=True
):
    # Capture
    cap = cv.VideoCapture(video_path)
    assert cap.isOpened()
    # cap_fourcc = int(cap.get(cv.CAP_PROP_FOURCC)) # may not work if codec is not supported.
    cap_fourcc = cv.VideoWriter_fourcc(*fourcc)
    cap_fps = cap.get(cv.CAP_PROP_FPS)
    cap_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    cap_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Save prepare
    if save_to:
        if os.path.isfile(save_to):
            print('{} already exist. Replace it? y/N'.format(save_to))
            do_replace = input() or 'n'
            if do_replace.lower() == 'n':
               print("k, bye.")
               exit(0)
            elif do_replace.lower() == 'y':
                pass
            else:
                print("You should type 'y' or 'n'.")
                exit(0)

        out = cv.VideoWriter()
        out.open(save_to, cap_fourcc, cap_fps, (cap_width, cap_height))

    # Detector
    anchor_path="babysister/YOLOv3_TensorFlow/data/yolo_anchors.txt"
    class_name_path="babysister/YOLOv3_TensorFlow/data/coco.names"
    restore_path="babysister/YOLOv3_TensorFlow/data/darknet_weights/yolov3.ckpt"
    yolov3 = YOLOv3(
        reversed(input_size), anchor_path, class_name_path, restore_path,
        max_boxes, score_thresh, iou_thresh)

    # Online multiple objects tracker
    tracker = SORTTracker()

    # Visulization stuff
    fontFace = cv.FONT_HERSHEY_SIMPLEX
    fontScale = 0.35
    fontThickness = 1
    boxThickness = 2
    if do_show:
        winname = 'Babysister. Input size {}'.format(input_size)
        cv.namedWindow(winname)
        # cv.moveWindow(winname, 0, 0)
        cv.waitKey(1)

    # info
    print('Processing {}'.format(video_path))
    print('YOLOv3 input size {}'.format(input_size))
    print('Max boxes: {}\nScore threshold: {}\nIOU threshold: {}'
        .format(max_boxes, score_thresh, iou_thresh))
    print('Result will be saved to {}\n'.format(save_to))

    # fps
    start_time = time.time()
    x = 1
    counter = 0
    fps = 0

    while cap.isOpened():
        # Frame info
        pos_frames = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        pos_msec = cap.get(cv.CAP_PROP_POS_MSEC)
        pos_S = int((pos_msec / 1000) % 60)
        pos_M = int((pos_msec / (1000 * 60)) % 60)
        pos_H = int((pos_msec / (1000 * 60 * 60)) % 24)
        pos_time = "{:02d}:{:02d}:{:02d}".format(pos_H, pos_M, pos_S)
        frame_info = "{}\nFrame: {}".format(pos_time, pos_frames)
        print(frame_info)

        # Read capture
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # input data
        input_data = cv.resize(
            frame, dsize=tuple(input_size), interpolation=cv.INTER_LANCZOS4)
        input_data = cv.cvtColor(input_data, cv.COLOR_BGR2RGB)
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

        # detect
        boxes, scores, labels = yolov3.detect(input_data)

        # filter by class
        if 'all' not in classes:
            tmp_boxes, tmp_scores, tmp_labels = [], [], []
            for box, score, label in zip(boxes, scores, labels):
                if yolov3.classes[label] in classes:
                    tmp_boxes.append(box)
                    tmp_scores.append(score)
                    tmp_labels.append(label)
            boxes, scores, labels = np.array(tmp_boxes), tmp_scores, tmp_labels

        # rescale boxes
        frame_h, frame_w = frame.shape[:2]
        if boxes.shape[0] > 0:
            size_ratio = np.divide([frame_w, frame_h], input_size)
            boxes[:,0] *= size_ratio[0]
            boxes[:,1] *= size_ratio[1]
            boxes[:,2] *= size_ratio[0]
            boxes[:,3] *= size_ratio[1]

        #filter by box size wrt image size.
        if np.greater([1,1], max_bb_size_ratio).any():
            tmp_boxes, tmp_scores, tmp_labels = [], [], []

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                size_ratio = np.divide([x2-x1,y2-y1], [frame_w, frame_h])

                if np.greater(size_ratio, max_bb_size_ratio).any():
                    continue

                tmp_boxes.append(box)
                tmp_scores.append(score)
                tmp_labels.append(label)

            boxes, scores, labels = np.array(tmp_boxes), tmp_scores, tmp_labels

        # track
        tracks = tracker.update(boxes, scores)

        # putText
        frame[:120, :150, 1] = 255
        text_x, text_y = 5, 25
        text_line_gap = 20

        # frame_info
        for i, line in enumerate(frame_info.split("\n")):
            text_y += i * text_line_gap
            cv.putText(frame, line, (text_x, text_y), fontFace, 0.5, (0,0,0), fontThickness)

        # fps
        str_fps = "FPS: {:.02f}".format(fps)
        text_y += text_line_gap
        cv.putText(frame, str_fps, (text_x, text_y), fontFace, 0.5, (0,0,0), fontThickness)
        print(str_fps)

        # counts
        str_counts = "Detected: {}\nTracked:  {}".format(len(labels), len(tracks))
        text_y += text_line_gap
        for i, line in enumerate(str_counts.split('\n')):
            text_y += i * text_line_gap
            cv.putText(frame, line, (text_x, text_y), fontFace, 0.5, (0,0,0), fontThickness)
        print(str_counts)

        # draw detections
        print('Detections:\n\tClass\tScore\tBox')
        for box, score, label in zip(boxes, scores, labels):
            color = create_unique_color_uchar(label) # (0,0,255)

            # box
            x1, y1, x2, y2 = map(int, box)
            cv.rectangle(frame, (x1,y1), (x2,y2), color, boxThickness)

            if do_show_class:
                # score
                putText_withBackGround(
                    frame, '{:.02f}'.format(score),
                    (x1,y1-20), fontFace, fontScale, fontThickness, color)

                # class
                putText_withBackGround(
                    frame, yolov3.classes[label],
                    (x1+40,y1-20), fontFace, fontScale, fontThickness, color)

            print('\t{}\t{}\t{}'.format(yolov3.classes[label], score, box))

        # draw tracking
        print('Tracking:\n\tID\tBox')
        for track in tracks:
            id_ = int(track[4])
            color = create_unique_color_uchar(id_)

            # box
            x1, y1, x2, y2 = map(int, track[0:4])
            cv.rectangle(frame, (x1,y1), (x2,y2), color, boxThickness)

            # id_
            putText_withBackGround(
                frame, str(id_), (x1,y1), fontFace, fontScale, fontThickness, color)

            print("\t{}\t{}".format(int(track[4]), track[0:4]))

        # save
        if save_to:
            assert [frame_w, frame_h] == [cap_width, cap_height]
            out.write(frame)

        # show
        if do_show:
            cv.imshow(winname, frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        # fps
        counter += 1
        if (time.time() - start_time) > x :
            fps = counter / (time.time() - start_time)
            counter = 0
            start_time = time.time()

        print(flush=True)

    cap.release()

    if save_to:
        out.release()

    if do_show:
        cv.destroyAllWindows()


def help():
    print("""
Objects detection and online tracking.

Usage:
    demo_video.py run VIDEO_PATH [INPUT_SIZE] [CLASSES]
                    [MAX_BOXES] [SCORE_THRESH] [IOU_THRESH] [MAX_BB_SIZE_RATIO]
                    [SAVE_TO] [FOURCC] [DO_SHOW] [DO_SHOW_CLASS]

    demo_video.py run --video-path VIDEO_PATH [--input-size INPUT_SIZE] [--classes CLASSES]
                [--max-boxes MAX_BOXES] [--score-thresh SCORE_THRESH] [--iou-thresh IOU_THRESH] [--max-bb-size-ratio MAX_BB_SIZE_RATIO]
                [--save-to SAVE_TO] [--fourcc FOURCC] [--do-show DO_SHOW] [--do-show-class DO_SHOW_CLASS]

Descriptions:
    --video-path <string>
        Path to input video.q

    --input-size <2-tuple>
        YOLOv3 input size.
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

    --max-bb-size-ratio <2-tuple, in range [0, 0] ~ [1, 1]>
        Boxes maximum size ratio wrt frame size.
        Default: [1,1]

    --save-to <string>
        Directory to save result images.
        Default: not to save

    --fourcc <4-string>
        FOURCC is short for "four character code"
        - an identifier for a video codec, compression format, color or pixel format used in media files.
        Default: "XVID"

    --do-show <boolean, or integer in range [0, 1]>
        Whether to display result.
        Default: True (1)

    --do-show-class <boolean, or integer in range [0, 1]>
        Whether to display class, score.
        Default: True (1)
    """)


if __name__ == '__main__':
    fire.Fire()
