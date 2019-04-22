from pprint import pprint
import glob
import time
import fire
import cv2 as cv
import numpy as np

from babysister.detector import YOLOv3
from babysister.YOLOv3_TensorFlow.utils.plot_utils import plot_one_box
from babysister.tracker import IOUtracker


def demo(
    frames_dir,
    input_size=(608,608),
    anchor_path="babysister/YOLOv3_TensorFlow/data/yolo_anchors.txt",
    class_name_path="babysister/YOLOv3_TensorFlow/data/coco.names",
    restore_path="babysister/YOLOv3_TensorFlow/data/darknet_weights/yolov3.ckpt"
):
    yolov3 = YOLOv3(input_size, anchor_path, class_name_path, restore_path)
    iou_tracker = IOUtracker(sigma_l=0, sigma_h=0.5, sigma_iou=0.5, t_min=2)

    frame_paths = sorted(glob.glob(frames_dir + '/*.jpg'))[4400:]
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
                tmp_boxes.append(box)
                tmp_scores.append(score)
                tmp_labels.append(label)
        boxes, scores, labels = np.array(tmp_boxes), tmp_scores, tmp_labels

        # rescale boxes
        size_ratio = np.divide(frame.shape[:2], input_size)
        boxes[:,0] *= size_ratio[1]
        boxes[:,1] *= size_ratio[0]
        boxes[:,2] *= size_ratio[1]
        boxes[:,3] *= size_ratio[0]

        # track
        tracks_active = iou_tracker.track(boxes, scores, frame_num)

        # draw
        for box, label in zip(boxes, labels):
            plot_one_box(
                frame, box, yolov3.classes[label],
                color=yolov3.color_table[label], line_thickness=None)

            cv.putText(
                frame, "FPS: {:.02f}".format(fps),
                (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        for id_, track in enumerate(tracks_active):
            x1, y1, x2, y2 = map(int, track['bboxes'][-1])
            cv.putText(
                frame, str(id_),
                (x1,y1+10), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # show
        cv.imshow('Babysister', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        # pprint(tracks_active)
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            print(i, yolov3.classes[label], score, box)
        print()

        # fps
        counter += 1
        if (time.time() - start_time) > x :
            fps = counter / (time.time() - start_time)
            counter = 0
            start_time = time.time()

    cv.destroyAllWindows()


if __name__ == '__main__':
    fire.Fire(demo)
