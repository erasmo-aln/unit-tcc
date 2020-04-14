# TrafficSignsClassifier
A traffic signs classifier using Tensorflow 2

General pipeline:

1) Create Dataset
    - Use tf.data, and TFRecords to upload

2) Create model with subclassing using GradientTape

3) Create losses and optimizers with subclassing

4) Create custom training loops and callbacks
