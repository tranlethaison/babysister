
demo = r"""
Objects detection and online tracking with multiple ROIs.

Usage:
demo.py run \
    FRAMES_DIR [ROIS_FILE] \
    [INPUT_SIZE] [CLASSES] \
    [MAX_BOXES] [SCORE_THRESH] [IOU_THRESH] [MAX_BB_SIZE_RATIO] \
    [SAVE_TO] [DO_SHOW] [DO_SHOW_CLASS]

demo.py run \
    --frames-dir FRAMES_DIR [--rois-file ROIS_FILE] \
    [--input-size INPUT_SIZE] [--classes CLASSES] \
    [--max-boxes MAX_BOXES] [--score-thresh SCORE_THRESH] \
    [--iou-thresh IOU_THRESH] [--max-bb-size-ratio MAX_BB_SIZE_RATIO] \
    [--save-to SAVE_TO] \
    [--do-show DO_SHOW] [--do-show-class DO_SHOW_CLASS]

Descriptions:
    --frames-dir <string>
        Directory that contain sequences of frames (jpeg).

    --rois-file <string>
        Path to ROIs file (created manualy, or with `select_rois.py`).
        If do not want to use ROIs, pass in an empty file.

        ROI contains: 
            top-left coordinate, width, height
        ROI format: 
            x y width height
        ROIs file contains ROI, each on 1 line. 

    --input-size <2-tuple>
        YOLOv3 input size.
        Pass in `None` to use frame sizes. (no resizing)
        Default: [416,416]

    --classes <list of string>
        list of classes used for filterring.
        Default: "['all']" (no filter)

    --max-boxes <integer>
        maximum number of predicted boxes you'd like.
        Default: 100

    --score-thresh <float, in range [0, 1]>
        if [ highest class probability score < score threshold]
            then get rid of the corresponding boxes
        Default: 0.5

    --iou-thresh <float, in range [0, 1]>
        "intersection over union" threshold used for NMS filtering.
        Default: 0.5

    --max_bb_size_ratio <2-tuple, in range [0, 0] ~ [1, 1]>
        Boxes maximum size ratio wrt frame size.
        Default: [1,1]

    --save-to <string>
        Directory to save result images.
        Default: not to save

    --do-show <boolean>
        Whether to display result.
        Default: True (1)

    --do-show-class <boolean>
        Whether to display classes, scores.
        Default: True (1)
    """
