import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from .YOLOv3_TF2.yolov3_tf2.models import YoloV3, YoloV3Tiny
from .YOLOv3_TF2.yolov3_tf2.dataset import transform_images


class YOLOv3TF2:
    """Wrapper for yolov3-tf2.

    Args:
        input_size (integer, optional): resize images to.
        score_thresh (float (from 0 to 1), optional): confidence score threshold.
        classes_path (str, optional): class names file path.
        weights_path (str, optional): path to folder that contains checkpoints.
        tiny (boolean, optional): yolov3 or yolov3-tiny.
        num_classes (integer, optional): number of classes in the model
    """

    def __init__(
        self,
        input_size=416,
        max_boxes=30,
        score_thresh=0.5,
        iou_thresh=0.5,
        tiny=False,
        num_classes=80,
        classes_path="babysister/YOLOv3_TF2/data/coco.names",
        weights_path="babysister/YOLOv3_TF2/checkpoints/yolov3.tf",
    ):
        self.input_size = input_size
        self.score_thresh = score_thresh

        physical_devices = tf.config.experimental.list_physical_devices("GPU")
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        if tiny:
            self.yolo = YoloV3Tiny(
                classes=num_classes,
                yolo_max_boxes=max_boxes,
                yolo_iou_threshold=iou_thresh,
                yolo_score_threshold=score_thresh,
            )
        else:
            self.yolo = YoloV3(
                classes=num_classes,
                yolo_max_boxes=max_boxes,
                yolo_iou_threshold=iou_thresh,
                yolo_score_threshold=score_thresh,
            )

        self.yolo.load_weights(weights_path).expect_partial()
        logging.info("weights loaded")

        self.class_names = [c.strip() for c in open(classes_path).readlines()]
        logging.info("classes loaded")

    def detect(self, img_np):
        """Returns detections of `input_data` image.

        Args:
            img_np (ndarray): numpy image (RGB) in format [width, height, channel].

        Returns:
            [boxes, scores, labels]:
                `boxes` (Tensor) is boxes coordinate in format [[x0, y0, x1, y1], ...].
                `scores` (Tensor) is confidence scores.
                `classes` (Tensor) is class indexes.
        """
        # img_raw = tf.convert_to_tensor(img_np, dtype=tf.uint8)
        # img_raw = img_np

        img = tf.expand_dims(img_np, 0)
        img = transform_images(img, self.input_size)

        t1 = time.time()
        boxes, scores, classes, nums = self.yolo.predict(img)
        t2 = time.time()
        # print("detect time: {}".format(t2 - t1))

        # logging.info("detections:")
        # for i in range(nums[0]):
        #     logging.info(
        #         "\t{}, {}, {}".format(
        #             self.class_names[int(classes[0][i])],
        #             np.array(scores[0][i]),
        #             np.array(boxes[0][i]),
        #         )
        #     )

        # Convert box measure from ratio to pixel
        boxes *= self.input_size

        return boxes[0], scores[0], classes[0]
