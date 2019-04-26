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
    frames_dir, input_size=(416,416),
    max_boxes=100, score_thresh=0.5, iou_thresh=0.5, max_size_ratio=(0.5,0.5),
    save_dir=None, do_show=True
):
    # Save prepare
    if save_dir:
        if os.path.isdir(save_dir):
            print('Save directory already exist. Replace it? y/N')
            do_replace = input() or 'n'
            if do_replace.lower() == 'n':
               print("k, bye.")
               exit(0)
            elif do_replace.lower() == 'y':
                pass
            else:
                print("You should type 'y' or 'n'.")
                exit(0)
        else:
            os.makedirs(save_dir)

    # Detector
    anchor_path="babysister/YOLOv3_TensorFlow/data/yolo_anchors.txt"
    class_name_path="babysister/YOLOv3_TensorFlow/data/coco.names"
    restore_path="babysister/YOLOv3_TensorFlow/data/darknet_weights/yolov3.ckpt"
    yolov3 = YOLOv3(
        input_size, anchor_path, class_name_path, restore_path,
        max_boxes, score_thresh, iou_thresh)

    # Online multiple objects tracker
    tracker = SORTTracker()

    # Frames sequence
    frames_path = sorted(glob.glob(frames_dir + '/*.jpg'))
    assert len(frames_path) > 0

    # Visulization stuff
    fontFace = cv.FONT_HERSHEY_SIMPLEX
    fontScale = 0.35
    fontThickness = 1
    boxThickness = 1
    if do_show:
        winname = 'Babysister. Input size {}'.format(input_size)
        cv.namedWindow(winname)
        cv.moveWindow(winname, 0, 0)
        cv.waitKey(1)

    # info
    print('Processing {} images from {}'.format(len(frames_path), frames_dir))
    print('YOLOv3 input size {}'.format(input_size))
    print('Max boxes: {}\nScore threshold: {}\nIOU threshold: {}'
        .format(max_boxes, score_thresh, iou_thresh))
    print('Result images will be saved to {}\n'.format(save_dir))

    # fps
    start_time = time.time()
    x = 1
    counter = 0
    fps = 0

    for frame_num, frame_path in enumerate(frames_path):
        print(frame_path)
        frame = cv.imread(frame_path, cv.IMREAD_COLOR)

        # input data
        input_data = cv.resize(frame, dsize=input_size, interpolation=cv.INTER_LANCZOS4)
        input_data = cv.cvtColor(input_data, cv.COLOR_BGR2RGB)
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

        # detect
        boxes, scores, labels = yolov3.detect(input_data)

        # filter by class
        tmp_boxes, tmp_scores, tmp_labels = [], [], []
        for box, score, label in zip(boxes, scores, labels):
            if yolov3.classes[label] == 'person':
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
        if np.greater([1,1], max_size_ratio).any():
            tmp_boxes, tmp_scores, tmp_labels = [], [], []

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                size_ratio = np.divide([x2-x1,y2-y1], [frame_w, frame_h])

                if np.greater(size_ratio, max_size_ratio).any():
                    continue

                tmp_boxes.append(box)
                tmp_scores.append(score)
                tmp_labels.append(label)

            boxes, scores, labels = np.array(tmp_boxes), tmp_scores, tmp_labels

        # track
        tracks = tracker.update(boxes, scores)

        # draw fps, counts
        frame[:95, :150, 1] = 255

        str_fps = "FPS: {:.02f}".format(fps)
        str_peoples = "Peoples being...\n detected: {}\n tracked:  {}" \
            .format(len(labels), len(tracks))

        cv.putText(
            frame, str_fps, (5, 25), fontFace, 0.5, (0,0,0), fontThickness)

        for i, line in enumerate(str_peoples.split('\n')):
            cv.putText(
                frame, line, (5, 45 + i * 20), fontFace, 0.5, (0,0,0), fontThickness)

        print(str_fps)
        print(str_peoples)

        # draw detections
        print('Detections:\n Box Score')
        for box, score in zip(boxes, scores):
            color = (0,0,255)
            # box
            x1, y1, x2, y2 = map(int, box)
            cv.rectangle(frame, (x1,y1), (x2,y2), color, boxThickness)

            # score
            putText_withBackGround(
                frame, '{:.02f}'.format(score),
                (x1,y1-20), fontFace, fontScale, fontThickness, color)

            print(' {} {}'.format(box, score))

        # draw tracking
        print('Tracking:\n ID Box')
        for track in tracks:
            id_ = int(track[4])
            color = create_unique_color_uchar(id_)

            # box
            x1, y1, x2, y2 = map(int, track[0:4])
            cv.rectangle(frame, (x1,y1), (x2,y2), color, boxThickness)

            # id_
            putText_withBackGround(
                frame, str(id_), (x1,y1), fontFace, fontScale, fontThickness, color)

            print(" {} {}".format(int(track[4]), track[0:4]))

        # save
        if save_dir:
            _, frame_name = os.path.split(frame_path)
            result_frame_path = os.path.join(save_dir, frame_name)
            cv.imwrite(result_frame_path, frame)

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

        print()

    if do_show:
        cv.destroyAllWindows()


def help():
    print("""
Peoples detection and online tracking.

Usage:
    demo.py run FRAMES_DIR [INPUT_SIZE] [MAX_BOXES] [SCORE_THRESH] [IOU_THRESH] [SAVE_DIR] [DO_SHOW]

    demo.py run --frames-dir FRAMES_DIR [--input-size INPUT_SIZE]
                [--max-boxes MAX_BOXES] [--score-thresh SCORE_THRESH] [--iou-thresh IOU_THRESH]
                [--save-dir SAVE_DIR] [--do-show DO_SHOW]

Descriptions:
    --frames-dir <string>
        Directory that contain sequences of frames (jpeg).

    --input-size <tuple>
        YOLOv3 input size.
        Default: '(416,416)'

    --max-boxes <integer.
        maximum number of predicted boxes you'd like.
        Default: 100

    --score-thresh <float, in [0, 1]>
        if [ highest class probability score < score threshold]
            then get rid of the corresponding boxes
        Default: 0.5

    --iou-thresh <float, in [0, 1]>
        "intersection over union" threshold used for NMS filtering.
        Default: 0.5

    --max_size_ratio [tuple, member in [0, 1]]
        Boxes maximum size ratio wrt frame size.

    --save-dir <string>
        Directory to save result images.
        Default: not to save

    --do-show <boolean>
        Whether to display result.
        Default: True
    """)


if __name__ == '__main__':
    fire.Fire()
