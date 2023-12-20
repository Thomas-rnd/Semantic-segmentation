# Semantic Segmentation

### Overview

This repository contains a comprehensive Semantic Segmentation Trainer designed for training and evaluating semantic segmentation models. The trainer is equipped with essential features such as model checkpointing, early stopping, and logging of key metrics.

Semantic segmentation is an important part of computer vision research, and since this domain is becoming a more central part of our lives, we need these technologies to be able to perform on even the smallest devices. In this work, we present a new approach to enhance the semantic segmentation efficiency of DeconvNet. Our main contribution is the replacement of the VGG16 backbone with MobileNetV3 small components. With this adjustment, the number of parameters in the model is reduced from 252 million to 12 million, and a non-proportional Intersection over Union (IoU) of about 50% is achieved. However, we also note difficulties in integrating MobileNetV3 small with the DeconvNet design.

### Features

1. Model Training: Train semantic segmentation models on custom datasets.
2. Model Checkpointing: Save model and optimizer checkpoints during training.
3. Early Stopping: Implement early stopping based on Mean Intersection over Union (mIoU) loss.
4. Logging with WandB: Log training progress, losses, and mIoU to Weights & Biases for easy monitoring.

### Dataset

![dataset](https://github.com/Thomas-rnd/Semantic-segmentation/blob/main/img/dataset.png)

### Results

![result](https://github.com/Thomas-rnd/Semantic-segmentation/blob/main/img/result.png)
