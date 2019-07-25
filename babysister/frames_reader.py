"""Frames reading routines"""
import glob

import cv2 as cv


class FrameReadError(Exception):
    """"""

    pass


class ImagesReader:
    """Reader for sequence of frames"""

    def __init__(self, frames_dir, im_format="jpg"):
        """"""
        self.frames_path = sorted(glob.glob(frames_dir + "/*." + im_format))
        assert len(self.frames_path) > 0

        self.current = 0

    def read(self):
        """"""
        try:
            frame = cv.imread(self.frames_path[self.current], cv.IMREAD_COLOR)
            self.current += 1
            return frame
        except IndexError as err:
            raise FrameReadError("Can't receive frame (stream end?)") from err

    def get_frame_size(self):
        """"""
        h, w, __ = cv.imread(self.frames_path[0], cv.IMREAD_COLOR).shape
        return w, h


class VideoReader:
    """Reader for video"""

    def __init__(self, video_path):
        """"""
        self.cap = cv.VideoCapture(video_path)
        assert self.cap.isOpened()

    def read(self):
        """"""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                raise FrameReadError("Can't receive frame (stream end?)")
            return frame

    def get_frame_size(self):
        """"""
        w = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        return w, h
