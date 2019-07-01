"""Frames reading routines"""
import glob

import cv2 as cv


class FramesReader:
    """Reader for sequence of frames"""

    def __init__(self, frames_dir, im_format="jpg"):
        self.frames_path = sorted(glob.glob(frames_dir + "/*." + im_format))
        if len(self.frames_path) == 0:
            print("WARNING: {} contains no images with format {}"
                .format(frames_dir, im_format))

    def read(self):
        for frame_path in self.frames_path: 
            yield cv.imread(frame_path, cv.IMREAD_COLOR)

    def get_frame_size(self):
        h, w, __ = cv.imread(self.frames_path[0], cv.IMREAD_COLOR)
        return w, h


class VideoReader:
    """Reader for video"""

    def __init__(self, video_path):
        self.cap = cv.VideoCapture(video_path)
        assert self.cap.isOpened()

    def read(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            yield frame

    def get_frame_size(self):
        w = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        return w, h

