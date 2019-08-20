import fire

from babysister import frames_reader
from babysister import runner


def frames(frames_dir, **kwargs):
    """Demo for sequence of frames"""
    framesReader = frames_reader.ImagesReader(frames_dir, "jpg")
    runner.run(framesReader, **kwargs)


def video(video_path, **kwargs):
    """Demo for video"""
    framesReader = frames_reader.VideoReader(video_path)
    runner.run(framesReader, **kwargs)


if __name__ == "__main__":
    fire.Fire()
