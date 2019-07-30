#rm -f source/babysister.*
sphinx-apidoc \
    -fe \
    -V 2.0.0 \
    -o ./source/ ../babysister/ \
    ../babysister/YOLOv3_TensorFlow/ \
    ../babysister/sort/

make clean
make html
