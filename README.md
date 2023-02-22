## DeepFake Detection

## Overview
  * [Intro](#intro)
  * [Requirements](#requirements)
  * [Usage](#usage)
  * [References](#references)


### Intro

The following is repo is a finetuning of a pre-trained EfficientNetv2 classification algorithm to learn to do DeepFake classification. Leveraged samples from mp4 from DeepFake MNIST+ dataset to transfer learn with our model. 


### Requirements
  * Pytorch
  * Pandas
  * Numpy
  * Matplotlib
  * CUDA 10+ highly recommended

### Usage

  * git clone this repo

  * cd into repo directory

  ```
  cd DeepFake Detection
  ```
  
  * create conda env

  ```
  conda env create -f env.yml
  ```

  * unzip images for classification to ./deepfake_detection/dataset folder such that there are 2 subdirectories "real" and "fake"
  
  * cd into deep fake and run efficient_net_b7.py
  ```
  cd deepfake_detection
  python -i efficient_net_b7.py
  ```

  * after training is complete

