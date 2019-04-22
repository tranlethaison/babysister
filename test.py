import glob
import time
import fire
import cv2 as cv
import numpy as np

from babysister.detector import YOLOv3
from babysister.YOLOv3_TensorFlow.utils.plot_utils import plot_one_box


def demo(
    frames_dir,
    input_size=(608,608),
    anchor_path="babysister/YOLOv3_TensorFlow/data/yolo_anchors.txt",
    class_name_path="babysister/YOLOv3_TensorFlow/data/coco.names",
    restore_path="babysister/YOLOv3_TensorFlow/data/darknet_weights/yolov3.ckpt"
):
    yolov3 = YOLOv3(input_size, anchor_path, class_name_path, restore_path)

    frame_paths = sorted(glob.glob(frames_dir + '/*.jpg'))
    assert len(frame_paths) > 0

    # fps
    start_time = time.time()
    x = 1
    counter = 0
    fps = 0

    for frame_path in frame_paths:
        frame = cv.imread(frame_path, cv.IMREAD_COLOR)

        # input data
        input_data = cv.resize(frame, dsize=input_size)
        input_data = cv.cvtColor(input_data, cv.COLOR_RGB2BGR)
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

        # detect
        boxes, scores, labels = yolov3.detect(input_data)

        # rescale boxes
        size_ratio = np.divide(frame.shape[:2], input_size)
        boxes[:,0] *= size_ratio[1]
        boxes[:,1] *= size_ratio[0]
        boxes[:,2] *= size_ratio[1]
        boxes[:,3] *= size_ratio[0]

        # draw
        for box, label in zip(boxes, labels):
            plot_one_box(
                frame, box, yolov3.classes[label],
                color=yolov3.color_table[label], line_thickness=None)

            cv.putText(
                frame, "FPS: {:.02f}".format(fps),
                (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # show
        cv.imshow('Detection', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

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
