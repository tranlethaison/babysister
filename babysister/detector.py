"""Wraper for YOLOv3_TensorFlow
"""
import tensorflow as tf

from .YOLOv3_TensorFlow.utils.misc_utils import parse_anchors, read_class_names
from .YOLOv3_TensorFlow.utils.nms_utils import gpu_nms
from .YOLOv3_TensorFlow.model import yolov3


class YOLOv3:
    def __init__(
        self, input_size, anchor_path, class_name_path, restore_path,
        max_boxes=30, score_thresh=0.5, iou_thresh=0.5
    ):
        self.input_size = input_size or (608, 608)
        self.anchor_path = anchor_path
        self.class_name_path = class_name_path
        self.restore_path = restore_path

        self.max_boxes = max_boxes
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh

        self.anchors = parse_anchors(self.anchor_path)
        self.classes = read_class_names(self.class_name_path)
        self.num_class = len(self.classes)

        self.sess = self.get_sess()

    def get_sess(self):
        """Detection session"""
        with tf.Graph().as_default() as g:
            # YOLOv3 graph
            self.input_data = tf.placeholder(
                tf.float32, (None,*self.input_size,3), 'input_data')

            yolo_model = yolov3(self.num_class, self.anchors)
            with tf.variable_scope('yolov3'):
                pred_feature_maps = yolo_model.forward(self.input_data, False)

            # predict
            pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
            pred_scores = pred_confs * pred_probs

            # non-maxima supression
            self.boxes, self.scores, self.labels = \
                gpu_nms(
                    pred_boxes, pred_scores, self.num_class,
                    self.max_boxes, self.score_thresh, self.iou_thresh)

            saver = tf.train.Saver()

        sess = tf.Session(graph=g)
        saver.restore(sess, self.restore_path)
        return sess

    def preprocess(self, input_data):
        return input_data / 255.0

    def detect(self, input_data):
        return self.sess.run(
            [self.boxes, self.scores, self.labels],
            feed_dict={self.input_data: self.preprocess(input_data)})


if __name__ == '__main__':
    import cv2 as cv
    import numpy as np
    from pprint import pprint

    input_size = (608, 608)
    anchor_path = r"YOLOv3_TensorFlow/data/yolo_anchors.txt"
    class_name_path = r"YOLOv3_TensorFlow/data/coco.names"
    restore_path = r"YOLOv3_TensorFlow/data/darknet_weights/yolov3.ckpt"
    yolov3 = YOLOv3(input_size, anchor_path, class_name_path, restore_path)

    img = cv.imread("joseph_redmon.jpg")
    input_data = cv.resize(img, dsize=input_size)
    input_data = cv.cvtColor(input_data, cv.COLOR_RGB2BGR)
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

    boxes, scores, labels = yolov3.detect(input_data)

    # rescale boxes
    size_ratio = np.divide(img.shape[:2], input_size)
    boxes[:,0] *= size_ratio[1]
    boxes[:,1] *= size_ratio[0]
    boxes[:,2] *= size_ratio[1]
    boxes[:,3] *= size_ratio[0]

    print('boxes:')
    pprint(boxes)
    print('scores:\n', scores)
    print('classes:\n', [yolov3.classes[l] for l in labels])

    for box, label in zip(boxes, labels):
        plot_one_box(
            img, box, yolov3.classes[label],
            color=yolov3.color_table[label], line_thickness=None)

    cv.imshow('YOLO v3', img)
    if cv.waitKey(0) & 0xFF == ord('q'):
        cv.destroyAllWindows()
