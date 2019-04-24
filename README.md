# Babysister

### 1. Introduction
A simple peoples detection and online tracking.

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

### 3. Weights convertion
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

### 5. Citing:
Awesome works that made this tool possible.

https://github.com/pjreddie/darknet

https://github.com/wizyoung/YOLOv3_TensorFlow

https://github.com/abewley/sort

@inproceedings{Bewley2016_sort,
  author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
  booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
  title={Simple online and realtime tracking},
  year={2016},
  pages={3464-3468},
  keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
  doi={10.1109/ICIP.2016.7533003}
}