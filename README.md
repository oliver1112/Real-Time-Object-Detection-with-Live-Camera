# Real-Time-Object-Detection-with-Live-Camera
Final project for deep learning course.

## Overview
This repo shows how to use Intel RealSense cameras with existing Deep Neural Network algorithms. The demo is similar to [MobileNet Single-Shot Detector](https://github.com/opencv/opencv/blob/3.4.0/samples/dnn/ssd_mobilenet_object_detection.cpp) provided with `opencv`. We modify it to work with Intel RealSense cameras and take advantage of depth data. 


The demo will load existing [Caffe model](https://github.com/chuanqi305/MobileNet-SSD) and use it to classify objects within the RGB image. Once object is detected, the demo will calculate approximate distance to the object using the depth data


To fine-tune a pre-trained MobileNet-SSD model on the KITTI dataset using Caffe, you'll need to follow a series of steps. This involves preparing the KITTI dataset, modifying the model and solver configuration files to suit the new dataset, and then running the training process. Here's a step-by-step guide along with a code template:

## Demo
![Demo](https://github.com/oliver1112/Real-Time-Object-Detection-with-Live-Camera/blob/main/demo/Screencast%20from%2012-15-2023%2011_54_58%20PM.gif)

## Instructions
### 1. Prepare the Dataset
- Convert the KITTI dataset annotations into a format compatible with SSD, usually the VOC format.
- Split the dataset into training and validation sets.

### 2. Modify the Model Configuration (Prototxt) Files
- **Training Prototxt (`train.prototxt`)**: Modify this file to ensure it is configured for training on the KITTI dataset. This involves adjusting the data layers to point to your KITTI dataset and ensuring the number of classes is set correctly for KITTI.
- **Solver Prototxt (`solver.prototxt`)**: This file contains settings for the training process (like learning rate, momentum, weight decay, etc.). Adjust these parameters as needed for fine-tuning.

### 3. Training and Result
For more training and experiment result, to see our [pdf](https://github.com/oliver1112/Real-Time-Object-Detection-with-Live-Camera/blob/main/Real-Time%20Object%20Detection%20with%20Live%20Camera%20.pdf).

### Important Notes:
- **Dataset Preparation**: Ensure your KITTI dataset is correctly formatted and the paths in the training prototxt file are correctly set.
- **Solver Configuration**: Fine-tuning typically requires a lower learning rate than training from scratch. Adjust the parameters in your solver prototxt accordingly.
- **Iterations and Display Interval**: These parameters control the duration of the training and how often you see updates. They may need to be adjusted based on the size of your dataset and the performance of your model.
- **Hardware Considerations**: Training deep learning models, especially on large datasets like KITTI, can be computationally intensive. Ensure you have appropriate hardware .

### Additional Considerations:
- **Backup and Monitor**: Regularly save your model during training and monitor its performance on a validation set.
- **Hyperparameter Tuning**: You might need to experiment with different hyperparameters to get the best results.
- **Compatibility**: Ensure that your version of Caffe supports SSD and the layers used in the MobileNet-SSD architecture.

This script provides a starting point for fine-tuning your pre-trained MobileNet-SSD model on the KITTI dataset. Depending on your specific setup and requirements, you may need to make additional adjustments or optimizations.


### Reference
https://github.com/chuanqi305/MobileNet-SSD
