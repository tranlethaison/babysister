# coding: utf-8
# for more details about the yolo darknet weights file, refer to
# https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe

from __future__ import division, print_function

import os
import sys
import tensorflow as tf
import numpy as np

from babysister.YOLOv3_TensorFlow.model import yolov3
from babysister.YOLOv3_TensorFlow.utils.misc_utils import parse_anchors, load_weights

num_class = 80
img_size = 416
weight_path = 'babysister/YOLOv3_TensorFlow/data/darknet_weights/yolov3.weights'
save_path = 'babysister/YOLOv3_TensorFlow/data/darknet_weights/yolov3.ckpt'
anchors = parse_anchors('babysister/YOLOv3_TensorFlow/data/yolo_anchors.txt')

model = yolov3(80, anchors)
with tf.Session() as sess:
    inputs = tf.placeholder(tf.float32, [1, img_size, img_size, 3])

    with tf.variable_scope('yolov3'):
        feature_map = model.forward(inputs)

    saver = tf.train.Saver(var_list=tf.global_variables(scope='yolov3'))

    load_ops = load_weights(tf.global_variables(scope='yolov3'), weight_path)
    sess.run(load_ops)
    saver.save(sess, save_path=save_path)
    print('TensorFlow model checkpoint has been saved to {}'.format(save_path))
