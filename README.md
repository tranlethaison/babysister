# Babysister

### 1. Introduction:
Peoples detection and online tracking.

### 2. Requirements:
```text
python=3
tensorflow >= 1.8.0 (lower versions may work too)
opencv-python
scikit-learn
scikit-image
filterpy
scipy
numba
```

### 3. Weights convertion:
The pretrained darknet weights file can be downloaded [here](https://pjreddie.com/media/files/yolov3.weights). Place this weights file under directory `babysister/YOLOv3_TensorFlow/data/darknet_weights/` and then run:

```shell
python convert_weight.py
```

Then the converted TensorFlow checkpoint file will be saved to the same directory.

You can also download the converted TensorFlow checkpoint file via [[Google Drive link](https://drive.google.com/drive/folders/1mXbNgNxyXPi7JNsnBaxEv1-nWr7SVoQt?usp=sharing)] or [[Github Release](https://github.com/wizyoung/YOLOv3_TensorFlow/releases/)]

### 4. Usage:
```shell
python demo.py help
```
```text
Peoples detection and online tracking.

Usage:
    demo.py run FRAMES_DIR [INPUT_SIZE] [MAX_BOXES] [SCORE_THRESH] [IOU_THRESH] [SAVE_DIR] [DO_SHOW]

    demo.py run --frames-dir FRAMES_DIR [--input-size INPUT_SIZE]
                [--max-boxes MAX_BOXES] [--score-thresh SCORE_THRESH] [--iou-thresh IOU_THRESH]
                [--save-dir SAVE_DIR] [--do-show DO_SHOW]

Descriptions:
    --frames-dir <string>
        Directory that contain sequences of frames (jpeg).

    --input-size <tuple>
        YOLOv3 input size.
        Default: '(416,416)'

    --max-boxes <integer.
        maximum number of predicted boxes you'd like.
        Default: 100

    --score-thresh <float, in [0, 1]>
        if [ highest class probability score < score threshold]
            then get rid of the corresponding boxes
        Default: 0.5

    --iou-thresh <float, in [0, 1]>
        "intersection over union" threshold used for NMS filtering.
        Default: 0.5

    --max_size_ratio [tuple, member in [0, 1]]
        Boxes maximum size ratio wrt frame size.

    --save-dir <string>
        Directory to save result images.
        Default: not to save

    --do-show <boolean>
        Whether to display result.
        Default: True
```

### 5. Credits:
Awesome works that made this tool possible.

https://github.com/pjreddie/darknet

https://github.com/wizyoung/YOLOv3_TensorFlow

https://github.com/abewley/sort
