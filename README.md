# Real-Time-Object-Detection-with-Live-Camera
Final project for deep learning course, the coding part


## Overview
This example shows how to use Intel RealSense cameras with existing [Deep Neural Network](https://en.wikipedia.org/wiki/Deep_learning) algorithms. The demo is similar to [MobileNet Single-Shot Detector](https://github.com/opencv/opencv/blob/3.4.0/samples/dnn/ssd_mobilenet_object_detection.cpp) provided with `opencv`. We modify it to work with Intel RealSense cameras and take advantage of depth data (in a very basic way). 


The demo will load existing [Caffe model](https://github.com/chuanqi305/MobileNet-SSD) (see another tutorial [here](https://docs.opencv.org/3.3.0/d5/de7/tutorial_dnn_googlenet.html) and use it to classify objects within the RGB image. Once object is detected, the demo will calculate approximate distance to the object using the depth data
