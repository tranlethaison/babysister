from pprint import pprint
import glob
import time
import fire
import cv2 as cv
import numpy as np

from babysister.detector import YOLOv3
from babysister.tracker import IOUtracker
from babysister.utils import create_unique_color_uchar

def demo(
    frames_dir,
    score_thresh=0.25,
    input_size=(608,608),
    anchor_path="babysister/YOLOv3_TensorFlow/data/yolo_anchors.txt",
    class_name_path="babysister/YOLOv3_TensorFlow/data/coco.names",
    restore_path="babysister/YOLOv3_TensorFlow/data/darknet_weights/yolov3.ckpt"
):
    yolov3 = YOLOv3(input_size, anchor_path, class_name_path, restore_path)
    iou_tracker = IOUtracker(sigma_l=0, sigma_h=0.5, sigma_iou=0.5, t_min=2)

    frame_paths = sorted(glob.glob(frames_dir + '/*.jpg'))
    assert len(frame_paths) > 0

    # fps
    start_time = time.time()
    x = 1
    counter = 0
    fps = 0

    for frame_num, frame_path in enumerate(frame_paths):
        frame = cv.imread(frame_path, cv.IMREAD_COLOR)

        # input data
        input_data = cv.resize(frame, dsize=input_size)
        input_data = cv.cvtColor(input_data, cv.COLOR_BGR2RGB)
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

        # detect
        boxes, scores, labels = yolov3.detect(input_data)

        # filter
        tmp_boxes, tmp_scores, tmp_labels = [], [], []
        for box, score, label in zip(boxes, scores, labels):
            if yolov3.classes[label] == 'person':
            # and score >= score_thresh:
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
        tracks = iou_tracker.track(boxes, scores, frame_num)

        # draw detections
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = map(int, box)
            cv.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)

            cv.putText(
                frame, "FPS: {:.02f}".format(fps), (30, 30), 
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        # draw tracking
        for id_, track in enumerate(tracks):
            color = create_unique_color_uchar(id_)
            # box
            x1, y1, x2, y2 = map(int, track['bboxes'][-1])
            cv.rectangle(frame, (x1,y1), (x2,y2), color, 2)

            # id_
            text_size = cv.getTextSize(str(id_), cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            center = x1 + 5, y1 + 5 + text_size[0][1]
            x2, y2 = x1 + 10 + text_size[0][0], y1 + 10 + text_size[0][1]

            cv.rectangle(frame, (x1,y1), (x2,y2), color, -1)
            cv.putText(
                frame, str(id_), center, 
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            pprint(track)
            print(len(track['bboxes']))

        print('frame_num', frame_num)
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            print(i, yolov3.classes[label], score, box)
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


if __name__ == '__main__':
    fire.Fire(demo)
