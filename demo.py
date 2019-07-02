"""Demo"""
import fire

from babysister import frames_reader
from babysister import runner


def frames(
    frames_dir, do_try_reading=False,
    rois_file='rois.csv',
    input_size=[416,416], valid_classes=['all'],
    max_boxes=100, score_thresh=0.5, iou_thresh=0.5, max_bb_size_ratio=[1,1],
    save_to=None, do_show=True, do_show_class=True,
    log_file='log.cvs'
):
    """Demo for sequence of frames"""
    runner.run(
        frames_reader.ImagesReader(frames_dir, "jpg"), do_try_reading,
        rois_file,
        input_size, valid_classes,
        max_boxes, score_thresh, iou_thresh, max_bb_size_ratio,
        save_to, do_show, do_show_class,
        log_file)


def video(
    video_path, do_try_reading=False,
    rois_file='rois.csv',
    input_size=[416,416], valid_classes=['all'],
    max_boxes=100, score_thresh=0.5, iou_thresh=0.5, max_bb_size_ratio=[1,1],
    save_to=None, do_show=True, do_show_class=True,
    log_file='log.cvs'
):
    """Demo for video"""
    runner.run(
        frames_reader.VideoReader(video_path), do_try_reading,
        rois_file,
        input_size, valid_classes,
        max_boxes, score_thresh, iou_thresh, max_bb_size_ratio,
        save_to, do_show, do_show_class,
        log_file)


if __name__ == '__main__':
    fire.Fire()

