"""Frames reading routines"""
import glob

import cv2 as cv


class FrameReadError(Exception):
    """Error that will be raised when a frame reader cannot read."""
    pass


class ImagesReader:
    """Reader for sequence of frames.

    Args:
        frames_dir (str): path to folder that contains frames.
        im_ext (str): image extension.

    Attributes:
        frames_path (list of str): 
            full path of images with `im_ext` extension in `frames_dir`.
    """

    def __init__(self, frames_dir, im_ext="jpg"):
        im_ext = im_ext.replace(".", "")
        self.frames_path = sorted(glob.glob(frames_dir + "/*." + im_ext))
        assert len(self.frames_path) > 0

        self.current = 0

    def read(self):
        """Try to read a frame from `frames_path`.

        Returns:
            ndarray: frame data in BGR format.

        Raises:
            FrameReadError: if cannot read.
        """
        try:
            frame = cv.imread(self.frames_path[self.current], cv.IMREAD_COLOR)
            self.current += 1
            return frame
        except IndexError as err:
            raise FrameReadError("Can't receive frame (stream end?)") from err

    def get_frame_size(self):
        """Get size of the first image in `frames_path`."""
        h, w, __ = cv.imread(self.frames_path[0], cv.IMREAD_COLOR).shape
        return w, h


class VideoReader:
    """Reader for video.

    Args:
        video_path (str): path to video.

    Attributes:
        cap (:py:class:`cv.VideoCapture`): VideoCapture of `video_path`.
    """

    def __init__(self, video_path):
        self.cap = cv.VideoCapture(video_path)
        assert self.cap.isOpened()

    def read(self):
        """Try to read a frame from `cap`.

        Returns:
            ndarray: frame data in BGR format.

        Raises:
            FrameReadError: if cannot read.
        """
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                raise FrameReadError("Can't receive frame (stream end?)")
            return frame

    def get_frame_size(self):
        """Get size of video."""
        w = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        return w, h
