# Babysister

- [Babysister](#babysister)
  - [Introduction](#introduction)
  - [Environment](#environment)
  - [Weights convertion](#weights-convertion)
  - [Usage](#usage)
    - [For video](#for-video)
    - [For sequence of frames](#for-sequence-of-frames)
  - [Demo](#demo)
  - [Credits](#credits)

## Introduction
Objects detection and online tracking.

## Environment
```text
python==3
tensorflow>=1.8.0
opencv-python
opencv-contrib-python
scikit-learn
scikit-image
filterpy
scipy
numba
fire
```

## Weights convertion
The pretrained darknet weights file can be downloaded [here](https://pjreddie.com/media/files/yolov3.weights). Place this weights file under directory `babysister/YOLOv3_TensorFlow/data/darknet_weights/` and then run:

```shell
$ python convert_weight.py
```

Then the converted TensorFlow checkpoint file will be saved to the same directory.

You can also download the converted TensorFlow checkpoint file via [[Google Drive link](https://drive.google.com/drive/folders/1mXbNgNxyXPi7JNsnBaxEv1-nWr7SVoQt?usp=sharing)] or [[Github Release](https://github.com/wizyoung/YOLOv3_TensorFlow/releases/)]

## Usage
### For video
```shell
$ python demo_video.py help
```
```text
Objects detection and online tracking.

Usage:
    demo_video.py run VIDEO_PATH [INPUT_SIZE] [CLASSES]
                    [MAX_BOXES] [SCORE_THRESH] [IOU_THRESH] [MAX_BB_SIZE_RATIO]
                    [SAVE_TO] [FOURCC] [DO_SHOW]

    demo_video.py run --video-path VIDEO_PATH [--input-size INPUT_SIZE] [--classes CLASSES]
                [--max-boxes MAX_BOXES] [--score-thresh SCORE_THRESH] [--iou-thresh IOU_THRESH] [--max-bb-size-ratio MAX_BB_SIZE_RATIO]
                [--save-to SAVE_TO] [--fourcc FOURCC] [--do-show DO_SHOW]

Descriptions:
    --video-path <string>
        Path to input video.q

    --input-size <2-tuple>
        YOLOv3 input size.
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

    --fourcc <4-string>
        FOURCC is short for "four character code"
        - an identifier for a video codec, compression format, color or pixel format used in media files.
        Default: "XVID"

    --do-show <boolean, or integer in [0, 1]>
        Whether to display result.
        Default: True (1)
```

### For sequence of frames
```shell
$ python demo.py help
```
```text
Objects detection and online tracking.

Usage:
    demo.py run FRAMES_DIR [INPUT_SIZE] [CLASSES]
                    [MAX_BOXES] [SCORE_THRESH] [IOU_THRESH] [MAX_BB_SIZE_RATIO]
                    [SAVE_TO] [DO_SHOW]

    demo.py run --frames-dir FRAMES_DIR [--input-size INPUT_SIZE] [--classes CLASSES]
                [--max-boxes MAX_BOXES] [--score-thresh SCORE_THRESH] [--iou-thresh IOU_THRESH] [--max-bb-size-ratio MAX_BB_SIZE_RATIO]
                [--save-to SAVE_TO] [--do-show DO_SHOW]

Descriptions:
    --frames-dir <string>
        Directory that contain sequences of frames (jpeg).

    --input-size <2-tuple>
        YOLOv3 input size.
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

    --do-show <boolean, or integer in [0, 1]>
        Whether to display result.
        Default: True (1)
```

## Demo
Download demo videos from [here](https://drive.google.com/drive/folders/1V5W7tBTlW9LoYb2HTenKWJp18eleh_TV?usp=sharing), place them in `demo` folder.

```shell
$ python demo_video.py run demo/TownCentre_720p.mp4 --input-size [640,360] --save-to demo/TownCentre_720p_result.mp4
```

## Credits
Awesome works that made this tool possible.

https://github.com/pjreddie/darknet

https://github.com/wizyoung/YOLOv3_TensorFlow

https://github.com/abewley/sort
