## Temporal Augmentation using frame-level features with RNN on UCF101
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Two-stream ConvNet has been recognized as one of the most deep ConvNet on video understanding, specifically human action recogniton. However, it suffers from the insufficient temporal datas for training. 

This repository aims to implement the temporal segments RNN for training on vidoes with temporal augmentation. The implementation is based on example code from [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch), and was largely modified in order to work with frame level features.

Pre-saved features generated from ResNet-101 is provided. 

#### Prerequisites
* Linux (tested on Ubuntu 14.04)
* [Torch](http://torch.ch/docs/getting-started.html#_)
* [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn)
* NVIDIA GPU is strongly recommended

#### Video Dataset 
[UCF101](http://crcv.ucf.edu/data/UCF101.php)

The start code provided here should be relatively easy to adapt for other dataset. For example:

- [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) 
- [Youtube-8M](https://research.google.com/youtube8m/index.html)

#### Features for training
I re-trained the two-stream ConvNet using pre-trained ResNet-101 on the UCF101 datasets. Please download the frame level features from the links below. 

**The features are coming soon.**

UCF-101 split 1
- [Spatial-steam and Temporal-steam ConvNet features](https://www.dropbox.com/s/ws34m5c1ah99xms/frameLevelFeaturesUCF101.tar.gz?dl=0) (34.7GB)

You can certainly generate features for split 2 and 3 by rearranging the features according the split list provided by UCF101. 

#### Usage
Specify the downloaded features and the types of RNN model you would like to use in `opt.lua`.
```
th main.lua
```

