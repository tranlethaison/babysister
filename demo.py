from pprint import pprint
import glob
import time
import fire
import cv2 as cv
import numpy as np

from babysister.detector import YOLOv3
from babysister.utils import create_unique_color_uchar
from babysister.tracker import SORTTracker


def run(frames_dir, input_size=(416,416)):
    # Detector
    anchor_path="babysister/YOLOv3_TensorFlow/data/yolo_anchors.txt"
    class_name_path="babysister/YOLOv3_TensorFlow/data/coco.names"
    restore_path="babysister/YOLOv3_TensorFlow/data/darknet_weights/yolov3.ckpt"
    yolov3 = YOLOv3(input_size, anchor_path, class_name_path, restore_path)

    # Online multiple objects tracker
    tracker = SORTTracker()

    # Frames sequence
    frame_paths = sorted(glob.glob(frames_dir + '/*.jpg'))
    assert len(frame_paths) > 0

    # fps
    start_time = time.time()
    x = 1
    counter = 0
    fps = 0

    for frame_num, frame_path in enumerate(frame_paths):
        print('Processing frame', frame_num)
        frame = cv.imread(frame_path, cv.IMREAD_COLOR)

        # input data
        input_data = cv.resize(frame, dsize=input_size, interpolation=cv.INTER_LANCZOS4)
        input_data = cv.cvtColor(input_data, cv.COLOR_BGR2RGB)
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

        # detect
        boxes, scores, labels = yolov3.detect(input_data)

        # filter
        tmp_boxes, tmp_scores, tmp_labels = [], [], []
        for box, score, label in zip(boxes, scores, labels):
            if yolov3.classes[label] == 'person':
                tmp_boxes.append(box)
                tmp_scores.append(score)
                tmp_labels.append(label)
        boxes, scores, labels = np.array(tmp_boxes), tmp_scores, tmp_labels

        # rescale boxes
        if boxes.shape[0] > 0:
            size_ratio = np.divide(frame.shape[:2], input_size)
            boxes[:,0] *= size_ratio[1]
            boxes[:,1] *= size_ratio[0]
            boxes[:,2] *= size_ratio[1]
            boxes[:,3] *= size_ratio[0]

        # track
        tracks = tracker.update(boxes, scores)

        # draw fps, counts
        str_fps = "FPS: {:.02f}".format(fps)
        str_peoples = "Peoples being...\n detected: {}\n tracked:  {}" \
            .format(len(labels), len(tracks))

        cv.putText(
            frame, str_fps, (30, 30),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        for i, line in enumerate(str_peoples.split('\n')):
            cv.putText(
                frame, line, (30, 50 + i * 20),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        print(str_fps)
        print(str_peoples)

        # draw detections
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)

        # draw tracking
        for track in tracks:
            x1, y1, x2, y2 = map(int, track[0:4])
            id_ = int(track[4])
            color = create_unique_color_uchar(id_)

            # box
            cv.rectangle(frame, (x1,y1), (x2,y2), color, 2)

            # id_
            text_size = cv.getTextSize(str(id_), cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            center = x1 + 5, y1 + 5 + text_size[0][1]
            x2_id, y2_id = x1 + 10 + text_size[0][0], y1 + 10 + text_size[0][1]

            cv.rectangle(frame, (x1,y1), (x2_id,y2_id), color, -1)
            cv.putText(
                frame, str(id_), center,
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            print("{} {}".format(int(track[4]), track[0:4]))

        # for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        #     print(i, yolov3.classes[label], score, box)
        print()

        # show
        cv.imshow('Babysister', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        # fps
        counter += 1
        if (time.time() - start_time) > x :
            fps = counter / (time.time() - start_time)
            counter = 0
            start_time = time.time()

    cv.destroyAllWindows()


def help():
    print("""
A simple peoples detection and online tracking.

Usage:
    demo.py run FRAMES_DIR [INPUT_SIZE] [ANCHOR_PATH] [CLASS_NAME_PATH] [RESTORE_PATH]

    demo.py run --frames-dir FRAMES_DIR [--input-size INPUT_SIZE]
                [--anchor-path ANCHOR_PATH] [--class-name-path CLASS_NAME_PATH] [--restore-path RESTORE_PATH]

Description:
    --frames-dir
        Directory that contain sequences of frames (jpeg).

    --input-size
        YOLOv3 input size. Default: '(416,416)'
    """)


if __name__ == '__main__':
    fire.Fire()
