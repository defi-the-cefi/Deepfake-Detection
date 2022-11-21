## RD-Data-Takehome-Anthony-Zelaya
DeepFake Detection

## Overview
  * [Intro](#intro)
  * [Requirements](#requirements)
  * [Usage](#usage)
  * [References](#references)


### Overview
The following is the completed Take Home exercises for Reality Defender Data Engineer.


### Requirements

  * Pytorch
  * CUDA 10+ highly recommended

### Usage

  * git clone this repo

  * cd into repo directory

  ```
  cd RD-Data-Takehome-Anthony-Zelaya
  ```
  
  * create conda env

  ```
  conda env create -f env.yml
  ```

  * unzip images for classification to ./deepfake_detection/dataset folder such that there are 2 subdirectories "real" and "fake"
  
  * cd into run efficient_net_b7.py
  ```
  cd deepfake_detection
  python -i efficient_net_b7.py
  ```
