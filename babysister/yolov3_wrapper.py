import tensorflow as tf

from .YOLOv3_TensorFlow.utils.misc_utils import parse_anchors, read_class_names
from .YOLOv3_TensorFlow.utils.nms_utils import gpu_nms
from .YOLOv3_TensorFlow.model import yolov3


_anchor_path = "babysister/YOLOv3_TensorFlow/data/yolo_anchors.txt"
_class_name_path = "babysister/YOLOv3_TensorFlow/data/coco.names"
_restore_path = "babysister/YOLOv3_TensorFlow/data/darknet_weights/yolov3.ckpt"


class YOLOv3:
    """Wrapper for YOLOv3_TensorFlow.

    Args:
        input_size (list of 2 integers, optional): input image size [width, height].
        max_boxes (int, optional): maximum detected bounding boxes.
        score_thresh (float (from 0 to 1), optional): confidence score threshold.
        iou_thresh (float (from 0 to 1), optional):
            IOU threshold use for detected boxes Non-Maximum Suppression.
        anchor_path (str): anchor file path.
        class_name_path (str): class names file path.
        restore_path (str): path to folder that contains checkpoints.
        session_config (:class:`tensorflow.ConfigProto` instance, optional):
            Detector session config.
    """
    def __init__(
        self,
        input_size=[416, 416],
        max_boxes=30,
        score_thresh=0.5,
        iou_thresh=0.5,
        anchor_path=_anchor_path,
        class_name_path=_class_name_path,
        restore_path=_restore_path,
        session_config=None,
    ):
        self.input_size = input_size
        self.max_boxes = max_boxes
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        self.anchor_path = anchor_path
        self.class_name_path = class_name_path
        self.restore_path = restore_path
        self.session_config = session_config

        self.anchors = parse_anchors(self.anchor_path)
        self.classes = read_class_names(self.class_name_path)
        self.num_class = len(self.classes)

        self.graph = tf.Graph()
        self.sess = self._get_sess()

    def _get_sess(self):
        """Returns yolov3 session restored from `restore_path`"""
        with self.graph.as_default():
            self.input_data = tf.placeholder(
                tf.float32, (None, *self.input_size, 3), "input_data"
            )

            yolo_model = yolov3(self.num_class, self.anchors)
            with tf.variable_scope("yolov3"):
                feature_maps = yolo_model.forward(self.input_data, is_training=False)

            # predict
            boxes, confs, probs = yolo_model.predict(feature_maps)
            scores = confs * probs

            # non-maxima supression
            self.boxes, self.scores, self.labels = gpu_nms(
                boxes,
                scores,
                self.num_class,
                self.max_boxes,
                self.score_thresh,
                self.iou_thresh,
            )

            saver = tf.train.Saver()
            sess = tf.Session(config=self.session_config)
            saver.restore(sess, self.restore_path)
            return sess

    def _preprocess(self, input_data):
        return input_data / 255.0

    def detect(self, input_data):
        """Returns detections of `input_data` image.

        Args:
            input_data (ndarray): image (RGB) in format [width, height, channel].

        Returns:
            [boxes, scores, labels]:
                `boxes` (Tensor) is boxes coordinate in format [[x0, y0, x1, y1], ...].
                `scores` (Tensor) is confidence scores.
                `labels` (Tensor) is label indexes.
        """
        return self.sess.run(
            [self.boxes, self.scores, self.labels],
            feed_dict={self.input_data: self._preprocess(input_data)},
        )
