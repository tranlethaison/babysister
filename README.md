_Work in progress_

# Babysister

- [Babysister](#babysister)
  - [Introduction](#introduction)
  - [Environment](#environment)
  - [Weights convertion](#weights-convertion)
  - [Usage](#usage)
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

## Demo
Download demo videos from [here](https://drive.google.com/drive/folders/1V5W7tBTlW9LoYb2HTenKWJp18eleh_TV?usp=sharing), place them in `demo` folder.

```shell
```

## Credits
Awesome works that made this tool possible.

https://github.com/pjreddie/darknet

https://github.com/wizyoung/YOLOv3_TensorFlow

https://github.com/abewley/sort
