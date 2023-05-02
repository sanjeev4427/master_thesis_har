# Reducing Human Activity Data Collection through Machine-Learned Activity-Wise Skips

## Introduction

Human Activity Recognition (HAR) technology is an area of great potential for improving our daily lives. With the increasing use of body wearable devices, people can monitor their fitness levels and avoid health issues. HAR also has the potential to play a major role in detecting and controlling pandemics, as well as diagnosing and monitoring various health issues. HAR has a wide range of applications in industries such as sports, production, logistics, gaming, surveillance, patient monitoring systems, and more. However, HAR devices face challenges due to their small size, limited battery life, and computational and storage capacity. In this thesis, the goal is to optimize the activity prediction process in a way to reduce energy consumption (resuced data collection) while minimizing the impact on the accuracy of the predictions. 

## Dataset

The following two datasets were used for the study, the Wetlab dataset and the RWHAR dataset. The Wetlab dataset is publicly available and contains sensor data recorded during a DNA extraction experiment. The data was collected from a 3D accelerometer worn on the wrist of the subjects and includes 8 activities such as cutting, pouring, and stirring, as well as a null class. On the other hand, the RWHAR dataset includes sensor data recorded from 7 on-body positions of 15 subjects who performed 8 different activities such as walking, running, and jumping. The data was recorded in natural settings, including jogging in a forest and climbing stairs in an old castle. Both datasets provide valuable information for developing machine learning algorithms that can recognize human activities.

## Training Pipeline

This thesis proposes a novel method to optimize the duty cycle of sensors through a combination of deep learning and machine learning techniques. By using predictions from a ConvLSTM model to train a machine learning model, the optimized duty cycle is designed to achieve maximum performance metrics while minimizing energy costs. During inference, the trained deep learning model is used to make activity predictions, and the corresponding duty cycle is used to control the on-off switching of the sensors. This optimization is implemented using two popular global-search heuristics algorithms: Simulated Annealing (SA) and Genetic Algorithm (GA). The deep learning training is done through a DeepConvLSTM architecture that outperforms other architectures proposed in the literature. 

<img src="/path/to/img.jpg" alt="Alt text" width="200" height="200" />

## Results

The results obtained for the RWHAR and wetlab datasets are shown below. The data saving of ... and drop in f1 score of ... is obtained. 
<img src="/path/to/img.jpg" alt="Alt text" width="200" height="200" />

## Acknowledgement

