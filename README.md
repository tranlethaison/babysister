# Babysister

-   [Babysister](#babysister)
    -   [Introduction](#introduction)
    -   [Environment](#environment)
    -   [Weights conversion](#weights-conversion)
    -   [Usage](#usage)
    -   [Demo](#demo)
    -   [TODO](#todo)
    -   [Credits](#credits)
    -   [Copying](#copying)

## Introduction

  Objects detection and online tracking.  
  Plug-in style code, can be easily adapted to different purposes.

* * *

## Environment

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

* * *

## Weights conversion

  Pre-trained darknet weights file can be downloaded [here](https://pjreddie.com/media/files/yolov3.weights).  
  Place weights file under directory  
  `babysister/YOLOv3_TensorFlow/data/darknet_weights/` and then run:

```shell
$ python convert_weight.py
```

  Converted TensorFlow checkpoint file will be saved to the same directory.  
  You can also download the converted TensorFlow checkpoint file via  
  [Google Drive link](https://drive.google.com/drive/folders/1mXbNgNxyXPi7JNsnBaxEv1-nWr7SVoQt?usp=sharing) or [Github Release](https://github.com/wizyoung/YOLOv3_TensorFlow/releases/)

* * *

## Usage

  [babysister/runner.py](babysister/runner.py): example usage

* * *

## Demo

  Download demo videos from [here](https://drive.google.com/drive/folders/1V5W7tBTlW9LoYb2HTenKWJp18eleh_TV?usp=sharing), place them in `demo` folder.  

  Select ROIs:

```shell
$ python select_rois.py demo/TownCentre_720p.mp4 --is-video 1 --save-to rois.csv
```

  Run:

```shell
$ python demo.py video demo/TownCentre_720p.mp4 \
    --rois-file rois.csv \
    --valid-classes ['person'] \
    --save-to frames/ \
    --log-file log.csv
```

* * *

## TODO

-   Add functions documentation.

* * *

## Credits

Awesome works that made this tool possible.  
<https://github.com/pjreddie/darknet>  
<https://github.com/wizyoung/YOLOv3_TensorFlow>  
<https://github.com/abewley/sort>

* * *

## Copying

  All my code is MIT licensed. Libraries follow their respective licenses.

* * *
